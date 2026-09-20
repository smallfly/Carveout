# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke test: gsplat rasterizes random Gaussians on this GPU.

First run also JIT-compiles gsplat's CUDA kernels (minutes; cached in
$TORCH_EXTENSIONS_DIR inside the project). Run on each rig:
    python scripts/smoke_test_gsplat.py
"""

import time

import torch


def main() -> None:
    import gsplat

    print(f"gsplat {gsplat.__version__}")
    device = "cuda"
    n = 100_000
    torch.manual_seed(0)

    means = torch.randn(n, 3, device=device) * 2.0
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
    scales = torch.rand(n, 3, device=device) * 0.02 + 0.005
    opacities = torch.rand(n, device=device)
    colors = torch.rand(n, 3, device=device)

    # Camera 8 units back, looking at the origin.
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 8.0
    W = H = 512
    K = torch.tensor(
        [[500.0, 0.0, W / 2], [0.0, 500.0, H / 2], [0.0, 0.0, 1.0]], device=device
    )

    print("rasterizing (first call JIT-compiles kernels)...")
    t0 = time.perf_counter()
    renders, alphas, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=W,
        height=H,
    )
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10):
        renders, alphas, _ = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmat[None], Ks=K[None], width=W, height=H,
        )
    torch.cuda.synchronize()
    t_steady = (time.perf_counter() - t0) / 10

    cov = (alphas > 0.01).float().mean().item()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    assert renders.shape == (1, H, W, 3), f"unexpected render shape {renders.shape}"
    assert cov > 0.05, f"render nearly empty (coverage {cov:.3f}) — kernels suspect"
    print(
        f"{n} gaussians @ {W}x{H}: first {t_first:.1f} s (incl. JIT), "
        f"steady {t_steady * 1e3:.1f} ms/frame | coverage {cov:.2f} | "
        f"peak VRAM {peak_mb:.0f} MB"
    )
    print("OK")


if __name__ == "__main__":
    main()
