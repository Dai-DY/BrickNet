#!/usr/bin/env python3
"""Fine-tune on car2brick data with BrickNet's own trainer (``scripts/train.py sft``).

The data from ``python -m car2brick build-data`` already has BrickNet's SFT layout (paths + captions jsonl), with
the descriptor text as the "caption". Car sequences are much longer than BrickNet's SFT models (hundreds of parts
plus anchor lines), so this wrapper only overrides the recipe's sequence length / iterations / batch size, then
hands over to ``train.main()``.

    python scripts/train_car2brick.py \
        --model Qwen/Qwen3-0.6B --pt_adapter kulits/BrickNet-0.6B-PT \
        --paths data/paths_train.jsonl --captions data/captions_train.jsonl \
        --val_paths data/paths_val.jsonl --val_captions data/captions_val.jsonl \
        --output runs/car2brick --max_len 12288 --iters 6000 --batch 4

Several training files (e.g. LEGO-derived + self-training rounds) can be concatenated first:
    cat data/paths_train.jsonl st1/paths_selftrain.jsonl > mix/paths.jsonl   (same for captions)
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--max_len", type=int, default=12288)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4, help="global batch size")
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--micro_batch", type=int, default=1,
                    help="sequences per forward pass; the rest of --batch is gradient accumulation")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing (needs much memory)")
    ap.add_argument("--no_compile", action="store_true", help="disable torch.compile (faster start, a bit slower)")
    known, rest = ap.parse_known_args()

    spec = importlib.util.spec_from_file_location("bricknet_train", Path(__file__).with_name("train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    # BrickNet's recipe puts the whole global batch on the device and keeps all activations, which is fine for its
    # ~2.8k-token SFT samples; car sequences reach ~12k tokens, so on one 16 GB card we need gradient checkpointing
    # and accumulation. Patched here instead of editing train.py.
    base_args = train.TrainingArguments

    def training_arguments(*a, **kw):
        micro = max(1, min(known.micro_batch, known.batch))
        kw["per_device_train_batch_size"] = kw["per_device_eval_batch_size"] = micro
        kw["gradient_accumulation_steps"] = max(1, known.batch // micro)
        if not known.no_grad_ckpt:
            kw["gradient_checkpointing"] = True
            kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if known.no_compile:
            kw["torch_compile"] = False
        return base_args(*a, **kw)

    train.TrainingArguments = training_arguments
    train.RECIPES["sft"].update(
        max_len=known.max_len,
        train_iters=known.iters,
        global_batch_size=known.batch,
        lr=known.lr,
        warmup_iters=known.warmup,
        eval_interval=known.eval_interval,
        save_interval=known.save_interval,
    )
    sys.argv = [sys.argv[0], "sft", *rest]
    train.main()


if __name__ == "__main__":
    main()
