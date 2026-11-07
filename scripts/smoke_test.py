"""TokJhalLM Environment and Hardware Smoke Test.

Validates that:
1. Base configuration schema (configs/base.yaml) contains all required keys.
2. CUDA is available, identifies GPU device name and total VRAM in GB.
3. Evaluates hardware bf16 support and falls back to fp16 on Turing (sm_75) GPUs.
4. Executes a GPU forward + backward pass using the effective precision.
5. Verifies bitsandbytes imports and creates an 8-bit AdamW optimizer on GPU.
6. Verifies wandb imports.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CONFIG_KEYS: dict[str, list[str]] = {
    "model": [
        "layers",
        "hidden",
        "heads",
        "vocab_size",
        "context_len",
        "tied_embeddings",
        "activation",
        "norm",
        "pos_enc",
        "dropout",
    ],
    "training": [
        "batch_size",
        "grad_accum",
        "lr",
        "warmup_steps",
        "weight_decay",
        "optimizer",
        "grad_clip",
        "precision",
        "gradient_checkpointing",
        "seed",
    ],
    "logging": [
        "wandb_project",
        "log_every",
        "ckpt_every",
        "val_every",
    ],
}


def validate_config(config_path: Path) -> dict[str, Any]:
    """Load and validate configs/base.yaml schema.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Loaded configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file is missing.
        ValueError: If any required section or key is missing.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing required configuration section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ValueError(
                    f"Missing required key '{key}' in section '{section}' of {config_path}"
                )

    return config


def verify_cuda() -> tuple[str, float, bool]:
    """Verify PyTorch CUDA availability, device name, VRAM, and bf16 support.

    Returns:
        Tuple containing (device_name, vram_gb, bf16_supported).

    Raises:
        RuntimeError: If CUDA is unavailable or no GPU is detected.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available. A CUDA-capable GPU is required.")

    device_name: str = torch.cuda.get_device_name(0)
    total_vram_bytes: int = torch.cuda.get_device_properties(0).total_memory
    vram_gb: float = total_vram_bytes / (1024**3)

    major, minor = torch.cuda.get_device_capability(0)
    # Hardware bf16 Tensor Cores require sm_80+ (Ampere or newer).
    # On Turing (sm_75, e.g. GTX 1650), hardware bf16 is not supported; fp16 is the native precision.
    hardware_bf16: bool = (major >= 8) and bool(torch.cuda.is_bf16_supported())

    return device_name, vram_gb, hardware_bf16


def verify_precision_forward_backward(bf16_supported: bool) -> str:
    """Run a forward and backward pass on GPU with the effective precision.

    Args:
        bf16_supported: Whether the GPU has native hardware bf16 acceleration.

    Returns:
        The precision name string used ('bf16' or 'fp16').
    """
    import torch
    import torch.nn as nn

    device = torch.device("cuda:0")
    if bf16_supported:
        effective_precision = "bf16"
        dtype = torch.bfloat16
    else:
        effective_precision = "fp16"
        dtype = torch.float16

    toy_model = nn.Sequential(
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)

    dummy_input = torch.randn(2, 16, device=device)

    with torch.amp.autocast("cuda", dtype=dtype):
        output = toy_model(dummy_input)
        loss = output.sum()

    loss.backward()

    return effective_precision


def verify_bitsandbytes() -> None:
    """Verify bitsandbytes 8-bit AdamW optimizer creation on GPU."""
    import bitsandbytes as bnb
    import torch

    device = torch.device("cuda:0")
    param = torch.nn.Parameter(torch.randn(4, 4, device=device, requires_grad=True))
    optimizer = bnb.optim.AdamW8bit([param], lr=1e-3)
    param.grad = torch.randn_like(param)
    optimizer.step()


def verify_wandb() -> None:
    """Verify wandb can be imported cleanly without login."""
    import wandb  # noqa: F401


def run_smoke_test() -> int:
    """Execute all smoke test checks sequentially.

    Returns:
        0 on success, non-zero on failure.
    """
    repo_root = Path(__file__).resolve().parent.parent
    config_file = repo_root / "configs" / "base.yaml"

    try:
        print("[1/5] Validating configs/base.yaml schema...")
        validate_config(config_file)
        print("      Config schema verified.")

        print("[2/5] Checking PyTorch CUDA and GPU hardware...")
        device_name, vram_gb, bf16_supported = verify_cuda()
        print(f"      GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
        print(f"      Hardware bf16 supported: {bf16_supported}")

        print("[3/5] Testing forward + backward pass under effective precision...")
        precision_used = verify_precision_forward_backward(bf16_supported)
        print(f"      Precision verification: {precision_used} forward+backward pass succeeded.")

        print("[4/5] Checking bitsandbytes 8-bit AdamW on GPU...")
        verify_bitsandbytes()
        print("      bitsandbytes AdamW8bit optimizer verified on GPU.")

        print("[5/5] Checking wandb importability...")
        verify_wandb()
        print("      wandb imported successfully.")

        print(
            f"SMOKE TEST PASSED — {device_name} / {vram_gb:.2f}GB / bf16={bf16_supported} ({precision_used} pass)"
        )
        return 0

    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
