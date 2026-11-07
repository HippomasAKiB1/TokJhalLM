"""fp32 vs fp16-autocast throughput and memory benchmark for the target GPU.

Evidence for CLAUDE.md "Hardware truths" and decision D1 (docs/decisions.md).
Times (1) raw GEMMs at the encoder's real layer shapes and (2) a full training
step (forward + backward + GradScaler + AdamW) of a stack shaped like the
configs/base.yaml encoder FFNs. All shapes come from configs/base.yaml.

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe scripts/bench_precision.py [--rounds N]

This is a proxy (no attention, no RoPE). Re-run the step benchmark on the real
encoder at Prompt 4 to confirm D1.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

WARMUP_ITERS = 5
TIMED_GEMM_ITERS = 30
TIMED_STEP_ITERS = 10


def load_base_config() -> dict[str, Any]:
    """Load configs/base.yaml from the repo root."""
    path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def gemm_tflops(m: int, k: int, n: int, dtype: Any) -> float:
    """Return achieved TFLOPS for an (m×k)@(k×n) matmul in the given dtype."""
    import torch

    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    for _ in range(WARMUP_ITERS):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(TIMED_GEMM_ITERS):
        a @ b
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / TIMED_GEMM_ITERS
    return 2 * m * k * n / seconds / 1e12


def train_step_ms(cfg: dict[str, Any], use_fp16: bool) -> tuple[float, float]:
    """Time one training step of an FFN stack shaped like the base encoder.

    Returns:
        (milliseconds per step, peak allocated MiB).
    """
    import torch
    import torch.nn as nn

    model_cfg, train_cfg = cfg["model"], cfg["training"]
    hidden, ffn = model_cfg["hidden"], model_cfg["ffn_hidden"]
    tokens = train_cfg["batch_size"] * model_cfg["context_len"]  # one micro-batch

    torch.manual_seed(train_cfg["seed"])
    model = nn.Sequential(
        *[
            nn.Sequential(nn.Linear(hidden, ffn), nn.GELU(), nn.Linear(ffn, hidden))
            for _ in range(model_cfg["layers"])
        ]
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    x = torch.randn(tokens, hidden, device="cuda")

    def step() -> None:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            loss = model(x).float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(WARMUP_ITERS):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(TIMED_STEP_ITERS):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / TIMED_STEP_ITERS * 1000
    return ms, torch.cuda.max_memory_allocated() / 2**20


def main() -> int:
    """Run the benchmark and print a comparison table."""
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=2, help="repeat count (exposes throttling)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; nothing to benchmark.", file=sys.stderr)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = load_base_config()
    m = cfg["model"]
    tokens = cfg["training"]["batch_size"] * m["context_len"]  # one micro-batch
    shapes = {
        "qkv/out proj": (tokens, m["hidden"], m["hidden"]),
        "ffn up": (tokens, m["hidden"], m["ffn_hidden"]),
        "ffn down": (tokens, m["ffn_hidden"], m["hidden"]),
        "vocab proj": (tokens, m["hidden"], m["vocab_size"]),
    }

    print(f"GPU: {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    for rnd in range(args.rounds):
        print(f"\n-- round {rnd + 1}/{args.rounds} --")
        for name, (mm, kk, nn_) in shapes.items():
            f32 = gemm_tflops(mm, kk, nn_, torch.float32)
            f16 = gemm_tflops(mm, kk, nn_, torch.float16)
            print(f"GEMM {name:<13} fp32 {f32:5.2f}  fp16 {f16:5.2f} TFLOPS  (fp16/fp32 {f16 / f32:.2f})")
        for use_fp16 in (False, True):
            ms, mib = train_step_ms(cfg, use_fp16)
            label = "fp16+GradScaler" if use_fp16 else "fp32"
            print(f"train step {label:<16} {ms:8.1f} ms   peak {mib:6.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
