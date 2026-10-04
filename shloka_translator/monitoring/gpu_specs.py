"""Spec-sheet lookup table for the GPUs Google Colab offers.

Figures are *peak* datasheet values (not measurements). Tensor throughput is
reported **dense** (without structured sparsity) wherever NVIDIA states the
sparse figure; this is noted per entry. Sources (NVIDIA datasheets/product
pages; checked when this table was built):

* T4  — https://www.nvidia.com/en-us/data-center/tesla-t4/ (2,560 CUDA, 320 Tensor, FP32 8.1 TFLOPS,
  FP16 65 TFLOPS, INT8 130 TOPS, 320+ GB/s, PCIe Gen3 x16, 70 W)
* L4  — https://www.nvidia.com/en-us/data-center/l4/ (FP32 30.3; FP16/BF16 242* and INT8 485* are
  *with sparsity*, "one-half lower without" → 121 / 242.5 dense; 300 GB/s; PCIe Gen4 x16; 72 W).
  CUDA/Tensor core counts (7,424 / 232) from the NVIDIA Ada GPU architecture whitepaper (AD104, 58 SMs).
* A100 — NVIDIA A100 datasheet (6,912 CUDA, 432 Tensor, FP32 19.5, BF16 312 dense, INT8 624 dense;
  40 GB HBM2 1,555 GB/s; 80 GB SXM 2,039 GB/s; 80 GB PCIe 1,935 GB/s)
* V100 — NVIDIA V100 datasheet (5,120 CUDA, 640 Tensor; SXM2 FP32 15.7 / Tensor 125; PCIe 14 / 112;
  900 GB/s). No INT8 Tensor Core figure is published.
* H100 — NVIDIA H100 datasheet (SXM: 16,896 CUDA, 528 Tensor, FP32 67, BF16 1,979* / INT8 3,958*
  with sparsity → 989.5 / 1,979 dense, 3.35 TB/s; PCIe: 14,592 CUDA, 456 Tensor, FP32 51,
  BF16 1,513* / INT8 3,026* sparse → 756.5 / 1,513 dense, 2.0 TB/s)
* RTX PRO 6000 Blackwell Server Edition (Colab "G4") — NVIDIA datasheet via Lenovo Press LP2263
  (24,064 CUDA, 752 Tensor, FP32 120 TFLOPS, FP16/BF16 Tensor "1 PFLOPS" peak — sparsity not stated,
  treat as an upper bound; 96 GB GDDR7, 512-bit, 1,597 GB/s; PCIe Gen5 x16; 600 W). INT8 not published.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUSpec:
    """Datasheet figures for one GPU model.

    Attributes:
        name: Canonical name.
        match: Lower-case substrings that identify it in the NVML name.
        cuda_cores: CUDA cores.
        tensor_cores: Tensor cores.
        fp32_tflops: FP32 peak TFLOPS.
        fp16_tflops: FP16/BF16 dense tensor peak TFLOPS.
        int8_tops: INT8 dense tensor peak TOPS (``None`` = not published).
        mem_bandwidth_gbs: Memory bandwidth GB/s.
        memory_gb: Memory size GB.
        pcie_gen: PCIe generation.
        pcie_width: PCIe lanes.
        tdp_w: Board power W.
        note: Caveats.
    """

    name: str
    match: tuple[str, ...]
    cuda_cores: int
    tensor_cores: int
    fp32_tflops: float
    fp16_tflops: float
    int8_tops: float | None
    mem_bandwidth_gbs: float
    memory_gb: float
    pcie_gen: int
    pcie_width: int
    tdp_w: float
    note: str = ""


# Ordered most specific first.
GPU_SPECS: tuple[GPUSpec, ...] = (
    GPUSpec(
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        ("rtx pro 6000",),
        24064,
        752,
        120.0,
        1000.0,
        None,
        1597,
        96,
        5,
        16,
        600,
        "FP16/BF16 '1 PFLOPS' peak as published; sparsity not stated (upper bound)",
    ),
    GPUSpec(
        "NVIDIA H100 SXM5 80GB",
        ("h100 80gb hbm3", "h100-sxm", "h100 sxm"),
        16896,
        528,
        67.0,
        989.5,
        1979.0,
        3350,
        80,
        5,
        16,
        700,
        "Tensor figures halved from NVIDIA's with-sparsity numbers",
    ),
    GPUSpec(
        "NVIDIA H100 PCIe 80GB",
        ("h100 pcie",),
        14592,
        456,
        51.0,
        756.5,
        1513.0,
        2000,
        80,
        5,
        16,
        350,
        "Tensor figures halved from NVIDIA's with-sparsity numbers",
    ),
    GPUSpec(
        "NVIDIA H100 (variant unknown, SXM figures)",
        ("h100",),
        16896,
        528,
        67.0,
        989.5,
        1979.0,
        3350,
        80,
        5,
        16,
        700,
        "Variant not recognised; SXM figures shown",
    ),
    GPUSpec(
        "NVIDIA A100 80GB SXM",
        ("a100-sxm4-80gb", "a100 sxm4 80gb"),
        6912,
        432,
        19.5,
        312.0,
        624.0,
        2039,
        80,
        4,
        16,
        400,
        "Dense tensor figures",
    ),
    GPUSpec(
        "NVIDIA A100 80GB PCIe",
        ("a100 80gb pcie", "a100-pcie-80gb"),
        6912,
        432,
        19.5,
        312.0,
        624.0,
        1935,
        80,
        4,
        16,
        300,
        "Dense tensor figures",
    ),
    GPUSpec(
        "NVIDIA A100 40GB",
        ("a100-sxm4-40gb", "a100-pcie-40gb", "a100"),
        6912,
        432,
        19.5,
        312.0,
        624.0,
        1555,
        40,
        4,
        16,
        400,
        "Dense tensor figures (SXM4 40 GB)",
    ),
    GPUSpec(
        "NVIDIA L4",
        ("nvidia l4", " l4"),
        7424,
        232,
        30.3,
        121.0,
        242.5,
        300,
        24,
        4,
        16,
        72,
        "Tensor figures halved from NVIDIA's with-sparsity numbers",
    ),
    GPUSpec(
        "NVIDIA Tesla T4",
        ("tesla t4", " t4"),
        2560,
        320,
        8.1,
        65.0,
        130.0,
        320,
        16,
        3,
        16,
        70,
        "FP16 = mixed-precision tensor peak",
    ),
    GPUSpec(
        "NVIDIA Tesla V100 PCIe",
        ("v100-pcie", "v100 pcie"),
        5120,
        640,
        14.0,
        112.0,
        None,
        900,
        16,
        3,
        16,
        250,
        "No INT8 tensor figure published",
    ),
    GPUSpec(
        "NVIDIA Tesla V100 SXM2",
        ("v100",),
        5120,
        640,
        15.7,
        125.0,
        None,
        900,
        16,
        3,
        16,
        300,
        "No INT8 tensor figure published",
    ),
)


def lookup_gpu_spec(gpu_name: str) -> GPUSpec | None:
    """Find the spec entry for an NVML GPU name.

    Args:
        gpu_name: e.g. ``"Tesla T4"`` or ``"NVIDIA A100-SXM4-40GB"``.

    Returns:
        :class:`GPUSpec` or ``None`` if the GPU is not in the table.

    Example:
        >>> lookup_gpu_spec("Tesla T4").cuda_cores
        2560
        >>> lookup_gpu_spec("NVIDIA L4").name
        'NVIDIA L4'
    """
    if not gpu_name:
        return None
    key = " " + gpu_name.lower().replace("nvidia ", " ").strip()
    full = gpu_name.lower()
    for spec in GPU_SPECS:
        if any(m in key or m in full for m in spec.match):
            return spec
    return None
