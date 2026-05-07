"""
Evaluate a Qwen3-VL checkpoint (or the base model) on a held-out slice of
CoSyn-400K. Used for the three-eval ablation:
    1. Base Qwen3-VL-2B (no training)               — zero-shot
    2. Task-only fine-tune (alignment_loss_weight=0)
    3. Task + alignment fine-tune (run1, weight=1e-5)

Reports exact-match (EM) and SQuAD-style token F1.

Held-out slice
--------------
The same val_split that train_qwen_cosyn.py uses for val_loss tracking
(test_size=0.02, seed=42, or raw["validation"] when present). It was never
trained on. The 200-example subsample is deterministic (subsample_seed=0)
so all three eval conditions score on identical inputs.

Run on GPU 0
------------
    CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_table_metrics \\
        --checkpoint_path outputs/qwen_cosyn_run1/best_model \\
        --tag run1 --num_eval_samples 200
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from datasets import load_dataset
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Metrics (SQuAD-style)
# ---------------------------------------------------------------------------

_PUNC_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _normalize(s: str) -> str:
    """Lowercase, strip articles, drop punctuation, collapse whitespace."""
    s = s.lower()
    s = _PUNC_RE.sub(" ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Held-out slice (mirrors train_qwen_cosyn.py exactly)
# ---------------------------------------------------------------------------

def build_eval_items(
    dataset_name: str,
    dataset_config: str,
    val_test_size: float,
    slice_seed: int,
    num_eval_samples: int,
    subsample_seed: int,
) -> List[Dict]:
    """
    Reproduce train_qwen_cosyn.py's val_split logic, then expand to (image,
    question, answer) items, then deterministically subsample to N items.
    """
    raw = load_dataset(dataset_name, dataset_config)
    if "validation" in raw:
        val_split = raw["validation"]
    else:
        split = raw["train"].train_test_split(
            test_size=val_test_size, seed=slice_seed
        )
        val_split = split["test"]

    items: List[Dict] = []
    for row_idx, row in enumerate(val_split):
        qa_pairs = row.get("qa_pairs", {}) or {}
        questions = qa_pairs.get("question", []) if isinstance(qa_pairs, dict) else []
        answers = qa_pairs.get("answer", []) if isinstance(qa_pairs, dict) else []
        if not questions or not answers:
            continue
        for qa_idx, (q, a) in enumerate(zip(questions, answers)):
            if not q or not a:
                continue
            items.append({
                "row_idx": row_idx,
                "qa_idx": qa_idx,
                "image": row["image"],
                "question": q,
                "answer": a,
            })

    if num_eval_samples > 0 and num_eval_samples < len(items):
        rng = np.random.RandomState(subsample_seed)
        chosen = sorted(rng.choice(len(items), size=num_eval_samples, replace=False).tolist())
        items = [items[i] for i in chosen]

    return items


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    model,
    processor,
    image,
    question: str,
    max_new_tokens: int,
    max_seq_len: int,
    device: torch.device,
) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    encoded = processor(
        text=[prompt_text],
        images=image_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(device)

    prompt_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
                or processor.tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, prompt_len:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve which weights to load: checkpoint dir if provided, else base.
    weights_path = args.checkpoint_path or args.model_name
    print(f"[eval] tag={args.tag}")
    print(f"[eval] loading processor from base ({args.model_name})")
    print(f"[eval] loading weights from: {weights_path}")

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        weights_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    print(f"[eval] building eval items from {args.dataset_name}/{args.dataset_config}")
    items = build_eval_items(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        val_test_size=args.val_test_size,
        slice_seed=args.slice_seed,
        num_eval_samples=args.num_eval_samples,
        subsample_seed=args.subsample_seed,
    )
    print(f"[eval] running on {len(items)} items")

    predictions: List[Dict] = []
    em_total = 0.0
    f1_total = 0.0
    t0 = time.time()

    for i, item in enumerate(items):
        try:
            pred = generate_answer(
                model=model,
                processor=processor,
                image=item["image"],
                question=item["question"],
                max_new_tokens=args.max_new_tokens,
                max_seq_len=args.max_seq_len,
                device=device,
            )
        except Exception as e:
            print(f"[eval] item {i} generation failed: {e}")
            pred = ""

        em = exact_match(pred, item["answer"])
        f1 = token_f1(pred, item["answer"])
        em_total += em
        f1_total += f1

        predictions.append({
            "row_idx": item["row_idx"],
            "qa_idx": item["qa_idx"],
            "question": item["question"],
            "gold": item["answer"],
            "pred": pred,
            "em": em,
            "f1": f1,
        })

        if (i + 1) % args.log_every == 0 or i == len(items) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"[eval] {i + 1}/{len(items)} "
                f"| EM={em_total / (i + 1):.3f} "
                f"| F1={f1_total / (i + 1):.3f} "
                f"| {rate:.2f} it/s"
            )

    n = len(items)
    em_score = em_total / n if n else 0.0
    f1_score = f1_total / n if n else 0.0
    elapsed = time.time() - t0

    summary = {
        "tag": args.tag,
        "checkpoint_path": args.checkpoint_path,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "num_items": n,
        "slice_seed": args.slice_seed,
        "val_test_size": args.val_test_size,
        "subsample_seed": args.subsample_seed,
        "exact_match": em_score,
        "token_f1": f1_score,
        "elapsed_sec": elapsed,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(summary, f, indent=2)
    pred_path = args.output_path.replace(".json", "_predictions.jsonl")
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"[eval] DONE — EM={em_score:.4f} F1={f1_score:.4f} on {n} items "
          f"in {elapsed:.1f}s")
    print(f"[eval] summary  → {args.output_path}")
    print(f"[eval] preds    → {pred_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Path to fine-tuned checkpoint dir. None = use base model.")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--dataset_name", type=str, default="allenai/CoSyn-400K")
    p.add_argument("--dataset_config", type=str, default="table")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_eval_samples", type=int, default=200,
                   help="0 = use full val_split. Default 200 for tractability.")
    p.add_argument("--slice_seed", type=int, default=42,
                   help="Must match train_qwen_cosyn val_split seed.")
    p.add_argument("--val_test_size", type=float, default=0.02,
                   help="Must match train_qwen_cosyn val_test_size.")
    p.add_argument("--subsample_seed", type=int, default=0)
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--output_path", type=str, required=True,
                   help="Where to save the summary JSON.")
    p.add_argument("--tag", type=str, required=True,
                   help="Label for this eval (e.g., 'base', 'task_only', 'run1').")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
