"""
train_qwen_cosyn.py

Custom training loop for fine-tuning Qwen3-VL on the CoSyn-400K table dataset.
Designed to be extensible for auxiliary alignment loss:
    loss = task_loss + w * alignment_loss
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
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
    # Placeholder for future auxiliary alignment loss.
    # When alignment_loss_weight > 0, a hook should populate alignment_loss
    # and total loss becomes: loss = task_loss + alignment_loss_weight * alignment_loss
    alignment_loss_weight: float = 0.0


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

class CoSynTableDataset(Dataset):
    """
    Each CoSyn-400K row contains one image and a list of qa_pairs.
    We expand so every (image, qa_pair) becomes one independent training item.
    This gives us maximum supervision signal per image.
    """

    def __init__(self, hf_split, processor, max_seq_len: int):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.items: List[Dict[str, Any]] = []

        for row in hf_split:
            image = row["image"]  # PIL Image
            qa_pairs = row.get("qa_pairs", [])
            if not qa_pairs:
                continue
            for qa in qa_pairs:
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                if not question or not answer:
                    continue
                self.items.append({
                    "image": image,
                    "question": question,
                    "answer": answer,
                })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
        # apply_chat_template with add_generation_prompt=True gives us exactly
        # the text up to where the model should start generating.
        messages_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        # Full text (user + assistant turn)
        full_text = self.processor.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Prompt-only text (user turn + generation prompt marker)
        prompt_text = self.processor.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process images via qwen_vl_utils — returns image tensors in the
        # format Qwen2.5-VL expects (pixel_values, image_grid_thw)
        image_inputs, _ = process_vision_info(messages_full)

        # Encode full sequence
        encoded = self.processor(
            text=[full_text],
            images=image_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = encoded["input_ids"].squeeze(0)           # (seq_len,)
        attention_mask = encoded["attention_mask"].squeeze(0) # (seq_len,)
        pixel_values = encoded["pixel_values"]                # (n_patches, C, H, W) — keep batch dim for cat
        image_grid_thw = encoded["image_grid_thw"]            # (n_images, 3) — keep batch dim for cat

        # --- Build labels: mask the prompt prefix with -100 ---
        # Tokenize the prompt-only text to get prompt length.
        # We re-use the same image so the visual tokens are counted correctly.
        prompt_encoded = self.processor(
            text=[prompt_text],
            images=image_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )
        prompt_len = prompt_encoded["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt tokens — only supervise on answer

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


# ---------------------------------------------------------------------------
# 3. Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    pixel_values and image_grid_thw are concatenated along dim 0 because
    Qwen2.5-VL expects a flat list of patches across the whole batch.
    Sequences are right-padded.
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
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


# ---------------------------------------------------------------------------
# 4. Validation
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
            outputs = model(**batch)
            # outputs.loss is mean over non-masked tokens in the batch
            non_masked = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * non_masked
            total_tokens += non_masked

    # Gather across processes if multi-GPU
    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
    total_tokens_tensor = torch.tensor(total_tokens, device=accelerator.device)
    total_loss_tensor = accelerator.reduce(total_loss_tensor, reduction="sum")
    total_tokens_tensor = accelerator.reduce(total_tokens_tensor, reduction="sum")

    avg_loss = (total_loss_tensor / total_tokens_tensor).item() if total_tokens_tensor > 0 else float("inf")
    model.train()
    return avg_loss


# ---------------------------------------------------------------------------
# 5. Main training loop
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
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    # --- Load dataset ---
    print(f"[train] Loading dataset: {config.dataset_name} (config={config.dataset_config})")
    raw = load_dataset(config.dataset_name, config.dataset_config)

    # CoSyn-400K may only have a 'train' split; create a small val split if needed
    if "validation" in raw:
        train_split = raw["train"]
        val_split = raw["validation"]
    else:
        split = raw["train"].train_test_split(test_size=0.02, seed=config.seed)
        train_split = split["train"]
        val_split = split["test"]

    print(f"[train] Train rows: {len(train_split)} | Val rows: {len(val_split)}")

    # --- Build datasets ---
    print("[train] Building CoSynTableDataset (expanding qa_pairs)...")
    train_dataset = CoSynTableDataset(train_split, processor, config.max_seq_len)
    val_dataset = CoSynTableDataset(val_split, processor, config.max_seq_len)
    print(f"[train] Train items (expanded): {len(train_dataset)} | Val items: {len(val_dataset)}")

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

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = (len(train_loader) // config.grad_accum_steps) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"[train] Total optimizer steps: {total_steps} | Warmup steps: {warmup_steps}")

    # --- Accelerator prepare ---
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(**batch)
                task_loss = outputs.loss

                # ----------------------------------------------------------------
                # ALIGNMENT LOSS HOOK
                # When alignment_loss_weight > 0, compute alignment_loss here
                # using intermediate representations (e.g., cross-modal cosine loss)
                # and combine:
                #     alignment_loss = compute_alignment_loss(model, batch, outputs)
                #     loss = task_loss + config.alignment_loss_weight * alignment_loss
                # ----------------------------------------------------------------
                alignment_loss = torch.tensor(0.0, device=task_loss.device)
                loss = task_loss + config.alignment_loss_weight * alignment_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().item()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % config.log_every == 0 and accelerator.is_main_process:
                    avg = running_loss / config.log_every
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"[train] epoch={epoch} step={global_step} "
                        f"loss={avg:.4f} lr={lr_now:.2e}"
                    )
                    running_loss = 0.0

        # --- Validation at end of each epoch ---
        print(f"[train] Running validation after epoch {epoch}...")
        val_loss = run_validation(model, val_loader, accelerator)

        if accelerator.is_main_process:
            print(f"[train] epoch={epoch} val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"{config.output_dir}/best_model"
                print(f"[train] New best val_loss={best_val_loss:.4f} — saving to {save_path}")
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_path)
                processor.save_pretrained(save_path)

    if accelerator.is_main_process:
        print(f"[train] Training complete. Best val_loss={best_val_loss:.4f}")
        print(f"[train] Best model saved to {config.output_dir}/best_model")


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL on CoSyn-400K table split")

    parser.add_argument("--model_name", type=str, default=Config.model_name)
    parser.add_argument("--dataset_name", type=str, default=Config.dataset_name)
    parser.add_argument("--dataset_config", type=str, default=Config.dataset_config)
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--num_epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=Config.warmup_ratio)
    parser.add_argument("--per_device_batch_size", type=int, default=Config.per_device_batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=Config.grad_accum_steps)
    parser.add_argument("--max_seq_len", type=int, default=Config.max_seq_len)
    parser.add_argument("--max_grad_norm", type=float, default=Config.max_grad_norm)
    parser.add_argument("--log_every", type=int, default=Config.log_every)
    parser.add_argument("--alignment_loss_weight", type=float, default=Config.alignment_loss_weight)

    args = parser.parse_args()
    config = Config(**vars(args))

    main(config)
