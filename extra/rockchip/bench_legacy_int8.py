#!/usr/bin/env python3
"""Run bench_llm with the superseded six-launch INT8 Linear instead of the fused one.

`NPULinear` now calls one C function per Linear. bench_llm has no switch for the old path and
adding one would leave a knob in the harness that exists only to reproduce a superseded
measurement, so this wrapper swaps the module-level function and then runs bench_llm's main.

Usage: DEV=CPU .venv/bin/python extra/rockchip/bench_legacy_int8.py \
           --ckpt <gguf> --name <name> --out <dir> --npu --quick --warmups 1
"""
from __future__ import annotations

import sys

from tinygrad import Tensor

import npu_linear
from npu_linear import packed_matmul_int8


def _six_launch(dense: Tensor, weight: Tensor, w_scale: Tensor, columns: int, device: str, m: int,
                row_offset: int = 0, k_offset: int = 0, column_offset: int = 0, cores: int = 3) -> Tensor:
    k = weight.shape[1] * 32
    return packed_matmul_int8(dense[row_offset:row_offset + m, k_offset:k_offset + k], weight, w_scale,
                              columns, device, cores, column_offset)


npu_linear.packed_matmul_int8_fused = _six_launch

import bench_llm  # noqa: E402

if __name__ == "__main__":
    sys.exit(bench_llm.main())
