"""
Calibrate alignment_loss_weight per issue #5.

Goal: find w such that  w * align_loss  approx  0.1 * task_loss
on items where alignment data is present.

Approach:
  1. Load N preprocessed .pt files
  2. Stream CoSyn-400K and grab the matching (image, qa_pairs) by row_id
  3. Run a forward pass through Qwen3-VL with the alignment hook on layer L
  4. Compute task CE loss (on answer tokens) and align MSE loss (mean-pool +
     fp32, matching compute_alignment_loss exactly)
  5. Average over items, print the ratio and recommended w

Run on GPU 0:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.calibrate_alignment --num-samples 20
"""

import argparse
import glob
import statistics

import torch
import torch.nn.functional as F

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

from src.train_qwen_cosyn import _image_hash
from src.token_map import (
    get_processed_resolution,
    scale_bbox,
    find_qwen3vl_image_tokens,
)


def main(num_samples: int, alignment_layer: int, max_seq_len: int) -> None:
    device = torch.device("cuda")
    model_name = "Qwen/Qwen3-VL-2B-Instruct"

    print(f"[calib] loading processor + model ({model_name})...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn not in env; SDPA is fine for calib
        trust_remote_code=True,
    ).to(device)
    model.model.visual.requires_grad_(False)
    model.eval()

    captured = {}

    def hook(module, inp, out):
        captured["hidden"] = out

    h = model.model.language_model.layers[alignment_layer].register_forward_hook(hook)
    print(f"[calib] hook on layer {alignment_layer}, visual frozen")

    # --- Load N alignment files and remember which row_ids we need ---
    pt_files = sorted(glob.glob("data/preprocessed/*.pt"))[:num_samples]
    print(f"[calib] using {len(pt_files)} preprocessed examples")
    needed = {}
    for f in pt_files:
        pt = torch.load(f, weights_only=False)
        needed[pt["row_id"]] = {
            "bboxes": pt["bboxes"],
            "text_reps": pt["text_hidden_states"][:, alignment_layer, :].clone(),
            "image_size": pt["image_size"],
            "hash": pt["image_hash"],
        }
    max_row = max(needed.keys())
    print(f"[calib] streaming CoSyn-400K up to row {max_row}...")

    # --- Stream HF dataset and collect matched rows ---
    dataset = load_dataset(
        "allenai/CoSyn-400K", "table", split="train", streaming=True
    )
    matched = []
    for stream_idx, sample in enumerate(dataset):
        if stream_idx > max_row:
            break
        if stream_idx not in needed:
            continue
        # qa_pairs is a dict-of-lists: {'question': [...], 'answer': [...], ...}
        qa_pairs = sample.get("qa_pairs", {}) or {}
        questions = qa_pairs.get("question", []) if isinstance(qa_pairs, dict) else []
        answers = qa_pairs.get("answer", []) if isinstance(qa_pairs, dict) else []
        if not questions or not answers:
            continue
        img_hash = _image_hash(sample["image"])
        assert img_hash == needed[stream_idx]["hash"], (
            f"hash mismatch at row {stream_idx}: "
            f".pt={needed[stream_idx]['hash']}, stream={img_hash}"
        )
        question, answer = questions[0], answers[0]
        if not question or not answer:
            continue
        matched.append({
            "image": sample["image"],
            "question": question,
            "answer": answer,
            "alignment": needed[stream_idx],
        })
    print(f"[calib] matched {len(matched)} usable examples")

    # --- Forward pass on each, compute both losses ---
    task_losses = []
    align_losses = []

    for i, item in enumerate(matched):
        image = item["image"]
        question = item["question"]
        answer = item["answer"]
        align = item["alignment"]

        messages_full = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer},
            ]},
        ]
        messages_prompt = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]

        full_text = processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        prompt_text = processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages_full)

        try:
            encoded = processor(
                text=[full_text],
                images=image_inputs,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            ).to(device)
            prompt_encoded = processor(
                text=[prompt_text],
                images=image_inputs,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
        except ValueError as e:
            # Image produces more tokens than max_seq_len allows; skip for calib
            print(f"[calib] {i+1}/{len(matched)}: skipping (oversize image: {e.args[0][:80]}...)")
            continue
        prompt_len = prompt_encoded["input_ids"].shape[1]

        labels = encoded["input_ids"].clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                mm_token_type_ids=encoded["mm_token_type_ids"],
                pixel_values=encoded["pixel_values"],
                image_grid_thw=encoded["image_grid_thw"],
                labels=labels,
            )
        task_loss = outputs.loss.item()

        # --- Alignment loss (mean-pool + fp32 MSE; matches compute_alignment_loss) ---
        layer_hidden = captured["hidden"]
        orig_w, orig_h = align["image_size"]
        processed_h, processed_w = get_processed_resolution(
            encoded["image_grid_thw"], image_index=0
        )
        seq_len = layer_hidden.shape[1]

        visual_reps = []
        text_reps = []
        for j, bbox in enumerate(align["bboxes"]):
            scaled = scale_bbox(bbox, orig_w, orig_h, processed_w, processed_h)
            tok_idxs = find_qwen3vl_image_tokens(
                encoded["input_ids"][0],
                encoded["image_grid_thw"],
                scaled,
                image_index=0,
            )
            valid = [idx for idx in tok_idxs if idx < seq_len]
            if not valid:
                continue
            visual_reps.append(layer_hidden[0, valid, :].mean(dim=0))
            text_reps.append(align["text_reps"][j])

        captured.clear()

        if not visual_reps:
            print(f"[calib] {i+1}/{len(matched)}: no valid visual tokens, skipping")
            continue

        v = torch.stack(visual_reps).float()
        t = torch.stack(text_reps).to(device).float()
        align_loss = F.mse_loss(v, t).item()

        task_losses.append(task_loss)
        align_losses.append(align_loss)
        print(
            f"[calib] {i+1}/{len(matched)}: "
            f"task={task_loss:.4f}  align={align_loss:.4f}  cells={len(visual_reps)}"
        )

    h.remove()

    if not task_losses:
        print("[calib] no usable items, nothing to report")
        return

    task_mean = statistics.mean(task_losses)
    align_mean = statistics.mean(align_losses)
    task_std = statistics.stdev(task_losses) if len(task_losses) > 1 else 0.0
    align_std = statistics.stdev(align_losses) if len(align_losses) > 1 else 0.0

    print("\n=== Calibration Summary ===")
    print(f"  examples used:    {len(task_losses)}")
    print(f"  mean task_loss:   {task_mean:.4f}  (std {task_std:.4f})")
    print(f"  mean align_loss:  {align_mean:.4f}  (std {align_std:.4f})")
    print(f"  ratio align/task: {align_mean/task_mean:.4f}")
    w = 0.1 * task_mean / align_mean
    print(f"  recommended alignment_loss_weight = 0.1 * task / align = {w:.6f}")
    print(f"    (so w * align_loss ~= 0.1 * task_loss on average)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--alignment-layer", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=2048)
    args = p.parse_args()
    main(args.num_samples, args.alignment_layer, args.max_seq_len)
