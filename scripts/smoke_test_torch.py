# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke test: torch cu128 sees the GPU and its kernels actually run.

Run on each rig (5090/sm_120 and 4090/sm_89): python scripts/smoke_test_torch.py
"""

import time

import torch


def main() -> None:
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device: {name} | sm_{cap[0]}{cap[1]} | {total_gb:.1f} GB")

    # A real kernel launch (not just device queries) — this is what fails when
    # the wheel lacks kernels for the GPU's arch.
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    assert torch.isfinite(c).all(), "matmul produced non-finite values"
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"4096^2 fp32 matmul: {dt * 1e3:.1f} ms | peak VRAM {peak_mb:.0f} MB")
    print("OK")


if __name__ == "__main__":
    main()
