"""
train_qwen_cosyn.py

Custom training loop for fine-tuning Qwen3-VL on the CoSyn-400K table dataset
with auxiliary alignment loss. The alignment loss encourages visual token
representations at a specific LM layer to match precomputed text-only
representations for the same table cell content.

    loss = task_loss + alignment_loss_weight * alignment_loss

Design decisions (per Sameen, 2026-04-22):
  - Raw L2 distance (MSE) on unnormalized hidden states, cast to fp32
  - Freeze vision encoder + MLP; only LM layers are trainable
  - Forward hook on a single layer (saves ~350MB vs output_hidden_states=True)
  - use_reentrant=False for gradient checkpointing (HF issue #21381)
"""

import argparse
import hashlib
from dataclasses import dataclass
from glob import glob as glob_files
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from qwen_vl_utils import process_vision_info

from src.token_map import (
    get_processed_resolution,
    scale_bbox,
    find_qwen3vl_image_tokens,
)


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    dataset_name: str = "allenai/CoSyn-400K"
    dataset_config: str = "table"
    output_dir: str = "outputs/qwen_cosyn_table"
    seed: int = 42
    num_epochs: int = 3
    lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    max_seq_len: int = 2048
    max_grad_norm: float = 1.0
    log_every: int = 50
    # Alignment loss config
    alignment_loss_weight: float = 0.01
    precomputed_dir: str = "data/preprocessed"
    alignment_layer: int = 16


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

def _image_hash(pil_image) -> str:
    """Short SHA256 hash of PIL image bytes for matching precomputed data."""
    return hashlib.sha256(pil_image.tobytes()).hexdigest()[:16]


class CoSynTableDataset(Dataset):
    """
    Each CoSyn-400K row contains one image and a list of qa_pairs.
    We expand so every (image, qa_pair) becomes one independent training item.

    If precomputed_dir is provided, loads alignment data (.pt files) and
    matches them to dataset rows via image hash.
    """

    def __init__(
        self,
        hf_split,
        processor,
        max_seq_len: int,
        precomputed_dir: Optional[str] = None,
        alignment_layer: int = 16,
    ):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.items: List[Dict[str, Any]] = []

        # --- Load precomputed alignment data ---
        self.hash_to_alignment: Dict[str, Dict[str, Any]] = {}
        if precomputed_dir:
            pt_files = sorted(glob_files(f"{precomputed_dir}/*.pt"))
            if pt_files:
                # Verify processor config on first file
                sample_pt = torch.load(pt_files[0], weights_only=False)
                pc = sample_pt["processor_config"]
                proc_min = getattr(processor.image_processor, "min_pixels", None)
                proc_max = getattr(processor.image_processor, "max_pixels", None)
                assert pc["min_pixels"] == proc_min, (
                    f"Processor min_pixels mismatch: .pt={pc['min_pixels']}, "
                    f"current={proc_min}"
                )
                assert pc["max_pixels"] == proc_max, (
                    f"Processor max_pixels mismatch: .pt={pc['max_pixels']}, "
                    f"current={proc_max}"
                )
                print("[dataset] Processor config verified against precomputed data")

                for pt_path in pt_files:
                    pt = torch.load(pt_path, weights_only=False)
                    self.hash_to_alignment[pt["image_hash"]] = {
                        "bboxes": pt["bboxes"],
                        # Only keep the layer we need: [num_cells, hidden_dim]
                        "text_reps": pt["text_hidden_states"][
                            :, alignment_layer, :
                        ].clone(),
                        "image_size": pt["image_size"],
                    }
                print(
                    f"[dataset] Loaded {len(self.hash_to_alignment)} "
                    f"precomputed alignment files"
                )

        # --- Expand rows into (image, qa_pair) items ---
        compute_hash = bool(self.hash_to_alignment)
        alignment_matches = 0

        for row in hf_split:
            image = row["image"]
            qa_pairs = row.get("qa_pairs", [])
            if not qa_pairs:
                continue

            img_hash = _image_hash(image) if compute_hash else ""
            if img_hash in self.hash_to_alignment:
                alignment_matches += 1

            for qa in qa_pairs:
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                if not question or not answer:
                    continue
                self.items.append({
                    "image": image,
                    "question": question,
                    "answer": answer,
                    "image_hash": img_hash,
                })

        if compute_hash:
            print(
                f"[dataset] {alignment_matches} images matched precomputed data"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        image = item["image"]
        question = item["question"]
        answer = item["answer"]

        # --- Build full conversation (user + assistant) ---
        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        # --- Build prompt-only conversation (for label masking) ---
        messages_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        full_text = self.processor.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process images via qwen_vl_utils
        image_inputs, _ = process_vision_info(messages_full)

        # Encode full sequence
        encoded = self.processor(
            text=[full_text],
            images=image_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        pixel_values = encoded["pixel_values"]
        image_grid_thw = encoded["image_grid_thw"]

        # --- Build labels: mask the prompt prefix with -100 ---
        prompt_encoded = self.processor(
            text=[prompt_text],
            images=image_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )
        prompt_len = prompt_encoded["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

        # --- Alignment data (if available for this image) ---
        img_hash = item["image_hash"]
        if img_hash in self.hash_to_alignment:
            align = self.hash_to_alignment[img_hash]
            result["has_alignment"] = True
            result["alignment_bboxes"] = align["bboxes"]
            result["alignment_text_reps"] = align["text_reps"]
            result["alignment_image_size"] = align["image_size"]
        else:
            result["has_alignment"] = False

        return result


# ---------------------------------------------------------------------------
# 3. Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Dict[str, Any]], pad_token_id: int
) -> Dict[str, Any]:
    """
    Collate model inputs (padded/concatenated tensors) and alignment metadata
    (passed through as Python lists).
    """
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.cat(
        [item["image_grid_thw"] for item in batch], dim=0
    )

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    # Alignment metadata — not tensors, passed through as lists
    result["has_alignment"] = [
        item.get("has_alignment", False) for item in batch
    ]
    result["alignment_bboxes"] = [
        item.get("alignment_bboxes", []) for item in batch
    ]
    result["alignment_text_reps"] = [
        item.get("alignment_text_reps", None) for item in batch
    ]
    result["alignment_image_size"] = [
        item.get("alignment_image_size", (0, 0)) for item in batch
    ]

    return result


# ---------------------------------------------------------------------------
# 4. Alignment loss computation
# ---------------------------------------------------------------------------

def compute_alignment_loss(
    layer_hidden: torch.Tensor,
    batch: Dict[str, Any],
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """
    Compute raw L2 (MSE) alignment loss between visual token representations
    at the hooked layer and precomputed text-only representations.

    For each image with alignment data:
      1. Scale OCR bboxes to the processor's resolution
      2. Map bboxes → visual token indices in the sequence
      3. Mean-pool hidden states at those positions → visual_rep
      4. Compare to precomputed text_rep via MSE in fp32

    Args:
        layer_hidden: Hidden states from hooked layer [batch, seq_len, hidden_dim].
        batch: Dict with alignment metadata (has_alignment, bboxes, text_reps, etc).
        image_grid_thw: Image grid tensor [batch, 3] for resolution computation.

    Returns:
        Scalar MSE loss in fp32, or 0.0 if no alignment data in this batch.
    """
    visual_reps = []
    text_reps = []

    for i in range(layer_hidden.shape[0]):
        if not batch["has_alignment"][i]:
            continue

        bboxes = batch["alignment_bboxes"][i]
        text_reps_i = batch["alignment_text_reps"][i]  # [num_cells, hidden_dim]
        orig_w, orig_h = batch["alignment_image_size"][i]

        # Processed resolution for this image
        processed_h, processed_w = get_processed_resolution(
            image_grid_thw[i : i + 1], image_index=0
        )

        input_ids_i = batch["input_ids"][i]  # [seq_len]
        seq_len = layer_hidden.shape[1]

        for j, bbox in enumerate(bboxes):
            scaled = scale_bbox(bbox, orig_w, orig_h, processed_w, processed_h)
            token_indices = find_qwen3vl_image_tokens(
                input_ids_i,
                image_grid_thw[i : i + 1],
                scaled,
                image_index=0,
            )

            # Safety: clamp to actual sequence length (padded sequences)
            valid_indices = [idx for idx in token_indices if idx < seq_len]
            if not valid_indices:
                continue

            # Mean-pool visual hidden states at this cell's token positions
            visual_tokens = layer_hidden[i, valid_indices, :]
            visual_rep = visual_tokens.mean(dim=0)

            visual_reps.append(visual_rep)
            text_reps.append(text_reps_i[j])

    if not visual_reps:
        return torch.tensor(0.0, device=layer_hidden.device, requires_grad=False)

    # Raw L2 (MSE) in fp32 for numerical stability
    visual_stack = torch.stack(visual_reps).float()
    text_stack = torch.stack(text_reps).to(visual_stack.device).float()

    return F.mse_loss(visual_stack, text_stack)


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

def run_validation(model, val_loader, accelerator) -> float:
    """
    Returns average loss weighted by the number of non-masked label tokens
    so batches of different lengths contribute proportionally.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            # Remove alignment metadata before passing to model
            for key in [
                "has_alignment",
                "alignment_bboxes",
                "alignment_text_reps",
                "alignment_image_size",
            ]:
                batch.pop(key, None)

            outputs = model(**batch)
            non_masked = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * non_masked
            total_tokens += non_masked

    # Gather across processes if multi-GPU
    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
    total_tokens_tensor = torch.tensor(total_tokens, device=accelerator.device)
    total_loss_tensor = accelerator.reduce(total_loss_tensor, reduction="sum")
    total_tokens_tensor = accelerator.reduce(total_tokens_tensor, reduction="sum")

    avg_loss = (
        (total_loss_tensor / total_tokens_tensor).item()
        if total_tokens_tensor > 0
        else float("inf")
    )
    model.train()
    return avg_loss


# ---------------------------------------------------------------------------
# 6. Main training loop
# ---------------------------------------------------------------------------

def main(config: Config):
    set_seed(config.seed)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.grad_accum_steps,
        project_dir=config.output_dir,
    )

    if accelerator.is_main_process:
        import os

        os.makedirs(config.output_dir, exist_ok=True)

    print(f"[train] Using device: {accelerator.device}")
    print(f"[train] Mixed precision: {accelerator.mixed_precision}")
    print(f"[train] Gradient accumulation steps: {config.grad_accum_steps}")

    # --- Load processor and model ---
    print(f"[train] Loading processor and model: {config.model_name}")
    processor = AutoProcessor.from_pretrained(
        config.model_name, trust_remote_code=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # --- Freeze vision encoder + MLP (per Sameen, 2026-04-22) ---
    # Only the LM layers (model.model.layers, embed_tokens, norm, lm_head) train.
    model.visual.requires_grad_(False)
    frozen_params = sum(p.numel() for p in model.visual.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Frozen (visual): {frozen_params:,} params")
    print(
        f"[train] Trainable (LM): {trainable_params:,} / {total_params:,} total"
    )

    # use_reentrant=False is required when some params are frozen (HF #21381)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # --- Register forward hook for alignment loss ---
    # Captures hidden states from a single LM layer without retaining all 28
    # layers (~350MB VRAM savings vs output_hidden_states=True).
    # Tensor is NOT detached — gradients flow through the LM layers.
    captured_hidden: Dict[str, torch.Tensor] = {}
    hook_handle = None

    if config.alignment_loss_weight > 0:

        def alignment_hook(module, input, output):
            # output is a tuple; output[0] is the hidden state tensor
            captured_hidden["hidden"] = output[0]

        hook_handle = model.model.layers[
            config.alignment_layer
        ].register_forward_hook(alignment_hook)
        print(
            f"[train] Alignment hook registered on layer {config.alignment_layer}"
        )
        print(f"[train] Alignment loss weight: {config.alignment_loss_weight}")

    # --- Load dataset ---
    print(
        f"[train] Loading dataset: {config.dataset_name} "
        f"(config={config.dataset_config})"
    )
    raw = load_dataset(config.dataset_name, config.dataset_config)

    if "validation" in raw:
        train_split = raw["train"]
        val_split = raw["validation"]
    else:
        split = raw["train"].train_test_split(
            test_size=0.02, seed=config.seed
        )
        train_split = split["train"]
        val_split = split["test"]

    print(f"[train] Train rows: {len(train_split)} | Val rows: {len(val_split)}")

    # --- Build datasets ---
    print("[train] Building CoSynTableDataset (expanding qa_pairs)...")
    precomputed = (
        config.precomputed_dir if config.alignment_loss_weight > 0 else None
    )
    train_dataset = CoSynTableDataset(
        train_split,
        processor,
        config.max_seq_len,
        precomputed_dir=precomputed,
        alignment_layer=config.alignment_layer,
    )
    val_dataset = CoSynTableDataset(val_split, processor, config.max_seq_len)
    print(
        f"[train] Train items (expanded): {len(train_dataset)} "
        f"| Val items: {len(val_dataset)}"
    )

    pad_id = processor.tokenizer.pad_token_id

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )

    # --- Optimizer (only trainable params) ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = (
        (len(train_loader) // config.grad_accum_steps) * config.num_epochs
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(
        f"[train] Total optimizer steps: {total_steps} "
        f"| Warmup steps: {warmup_steps}"
    )

    # --- Accelerator prepare ---
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_task_loss = 0.0
        running_align_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                # Separate alignment metadata before passing to model
                has_alignment = batch.pop("has_alignment")
                alignment_bboxes = batch.pop("alignment_bboxes")
                alignment_text_reps = batch.pop("alignment_text_reps")
                alignment_image_size = batch.pop("alignment_image_size")

                # Keep a copy for alignment computation (before model consumes it)
                image_grid_thw_raw = batch["image_grid_thw"].clone()

                # Forward pass — hook captures layer hidden states
                outputs = model(**batch)
                task_loss = outputs.loss

                # --- Alignment loss ---
                if (
                    config.alignment_loss_weight > 0
                    and "hidden" in captured_hidden
                    and any(has_alignment)
                ):
                    align_batch = {
                        "has_alignment": has_alignment,
                        "alignment_bboxes": alignment_bboxes,
                        "alignment_text_reps": alignment_text_reps,
                        "alignment_image_size": alignment_image_size,
                        "input_ids": batch["input_ids"],
                    }
                    alignment_loss = compute_alignment_loss(
                        captured_hidden["hidden"],
                        align_batch,
                        image_grid_thw_raw,
                    )
                    captured_hidden.clear()
                else:
                    alignment_loss = torch.tensor(0.0, device=task_loss.device)
                    captured_hidden.clear()

                loss = task_loss + config.alignment_loss_weight * alignment_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_task_loss += task_loss.detach().item()
            running_align_loss += alignment_loss.detach().item()

            if accelerator.sync_gradients:
                global_step += 1

                if (
                    global_step % config.log_every == 0
                    and accelerator.is_main_process
                ):
                    avg_task = running_task_loss / config.log_every
                    avg_align = running_align_loss / config.log_every
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"[train] epoch={epoch} step={global_step} "
                        f"task_loss={avg_task:.4f} "
                        f"align_loss={avg_align:.4f} "
                        f"lr={lr_now:.2e}"
                    )
                    running_task_loss = 0.0
                    running_align_loss = 0.0

        # --- Validation at end of each epoch ---
        print(f"[train] Running validation after epoch {epoch}...")
        val_loss = run_validation(model, val_loader, accelerator)

        if accelerator.is_main_process:
            print(f"[train] epoch={epoch} val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"{config.output_dir}/best_model"
                print(
                    f"[train] New best val_loss={best_val_loss:.4f} "
                    f"— saving to {save_path}"
                )
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_path)
                processor.save_pretrained(save_path)

    # --- Cleanup ---
    if hook_handle is not None:
        hook_handle.remove()

    if accelerator.is_main_process:
        print(f"[train] Training complete. Best val_loss={best_val_loss:.4f}")
        print(f"[train] Best model saved to {config.output_dir}/best_model")


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL on CoSyn-400K table split"
    )

    parser.add_argument("--model_name", type=str, default=Config.model_name)
    parser.add_argument(
        "--dataset_name", type=str, default=Config.dataset_name
    )
    parser.add_argument(
        "--dataset_config", type=str, default=Config.dataset_config
    )
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--num_epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument(
        "--weight_decay", type=float, default=Config.weight_decay
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=Config.warmup_ratio
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=Config.per_device_batch_size,
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=Config.grad_accum_steps
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=Config.max_seq_len
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=Config.max_grad_norm
    )
    parser.add_argument("--log_every", type=int, default=Config.log_every)
    parser.add_argument(
        "--alignment_loss_weight",
        type=float,
        default=Config.alignment_loss_weight,
    )
    parser.add_argument(
        "--precomputed_dir", type=str, default=Config.precomputed_dir
    )
    parser.add_argument(
        "--alignment_layer", type=int, default=Config.alignment_layer
    )

    args = parser.parse_args()
    config = Config(**vars(args))

    main(config)
