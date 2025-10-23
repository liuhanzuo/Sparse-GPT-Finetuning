"""
Minimal runner for fsdp_finetune with safe defaults.

This script lets you quickly test the training pipeline without editing the
original files. It runs a very short training loop on facebook/opt-125m.

Notes:
- Set `sparsity=1.0` to skip loading any pruned checkpoint.
- Increase `num_epochs`, `train_steps`, and `max_step` for real runs.
"""

from fsdp_finetune import fsdp_finetune


if __name__ == "__main__":
    config = {
        "lr": 2e-5,
        # keep small for a quick smoke test; raise for real training
        "num_epochs": 1,
        "seed": 1,
        # batch size > 1 may require more GPU memory
        "batch_size": 1,
        # model choices seen in this repo include: opt-125m, opt-350m, opt-1.3b
        "model_name": "opt-125m",
        # 1.0 = no pruning checkpoint to load
        "sparsity": 1.0,
        # scheduler steps; keep in sync with max_step for short runs
        "train_steps": 50,
        # break early in the inner loop to keep runs short
        "max_step": 50,
        # save the full model after training; set False for quick tests
        "save_model": False,
    }
    fsdp_finetune(config)
