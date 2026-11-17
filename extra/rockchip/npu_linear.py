"""Minimal RK3588 GEMM offload: one NPU matmul per Linear, no fused kernels.

This is the baseline path: every nn.Linear is replaced by the same generic FP16
GEMM, everything else stays on the CPU device. Layouts match ops_rockchip:
lhs (m,k) -> (k/8,m,8), weight (n,k) -> (n/16,k/32,16,32), out (n/4,m,4).

Packed weights are spread across IOMMU domains when they do not fit in one, so a
model larger than the ~4 GB per-domain IOVA window can still be loaded.
"""
from __future__ import annotations

import struct
from dataclasses import replace

from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import Target, round_up
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

TILE = 24576  # max packed columns per weight tile
K_TILE = 4096  # max K per matmul call (CBUF limit)

# Bytes of packed weights per IOMMU domain. The driver gives each domain an
# aperture of exactly 4 GiB (rknpu_iommu.c: dma_limit = 1ULL << 32), but this
# budget counts weights only: command buffers, activation buffers and
# allocator-retained frees are not in the sum. 3.906 GiB was tried and reverted —
# it bought nothing measurable (Qwen3.5 2B 2.09/3.05 vs 2.05/3.04 tok/s) and one
# Gemma2 2B run ended in a board reboot. Measured: 3.76 GB of packed weights fits
# in one domain and 5.23 GB does not. 3 GiB matches the reference's default.
DOMAIN_BUDGET = 3 << 30


class _Domains:
    """Assign each Linear's packed weights to an IOMMU domain, in walk order.

    Walk order is block order, so a block's weights share a domain unless that
    block straddles a budget boundary, and the NPU switches domain a few times
    per token instead of once per Linear.
    """

    def __init__(self) -> None:
        self.used: list[int] = [0]

    def place(self, nbytes: int) -> str:
        if nbytes > DOMAIN_BUDGET:
            raise MemoryError(f"Linear of {nbytes} bytes exceeds the {DOMAIN_BUDGET}-byte domain budget")
        if self.used[-1] + nbytes > DOMAIN_BUDGET:
            if len(self.used) >= 16:
                raise MemoryError("16 NPU IOMMU domains exhausted")
            self.used.append(0)
        self.used[-1] += nbytes
        return "ROCKCHIP" if len(self.used) == 1 else f"ROCKCHIP:{len(self.used) - 1}"


# One allocator per process: the IOVA window belongs to the driver domain, so a
# second model must not start booking domain 0 from zero again.
_DOMAINS = _Domains()


def _matmul(out: UOp, lhs: UOp, rhs: UOp, m: int, n: int, k: int, cores: int, column_offset: int) -> UOp:
    sink = UOp.sink(out, lhs, rhs, arg=KernelInfo(f"rockchip_matmul_{m}_{n}_{k}"))
    info = replace(ProgramInfo.from_sink(sink, Target("ROCKCHIP")), outs=(0,), ins=(1, 2))
    return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=(out, lhs, rhs)),
               UOp(Ops.SOURCE, arg=f"Rockchip FP16 {m}x{k}x{n}"),
               UOp(Ops.BINARY, arg=struct.pack("<7I", m, n, k, cores, column_offset * k * 2, 0, 0))))


def pack_weight(weight: Tensor, device: str) -> Tensor:
    n, k = weight.shape
    n, k = round_up(n, 32), round_up(k, 32)
    return weight.half().pad(((0, n - weight.shape[0]), (0, k - weight.shape[1]))).reshape(n // 16, 16, k // 32, 32) \
        .permute(0, 2, 1, 3).contiguous().to(device)


def packed_matmul(x: Tensor, weight: Tensor, columns: int, device: str, cores: int = 3, column_offset: int = 0) -> Tensor:
    m, original_k = x.shape
    k = weight.shape[1] * 32
    n = round_up(columns, 32)
    lhs = x.half().pad(((0, 0), (0, k - original_k))).reshape(m, k // 8, 8).permute(1, 0, 2).contiguous().to(device)
    out = Tensor.empty(n // 4, m, 4, device=device, dtype=dtypes.float)
    out = out.custom_kernel(lhs, weight, fxn=lambda o, a, w: _matmul(o, a, w, m, n, k, cores, column_offset))[0]
    return out.to(x.device).permute(1, 0, 2).reshape(m, n)[:, :columns]


class NPULinear:
    """Drop-in nn.Linear replacement; weights packed once, rows, columns and K tiled."""

    def __init__(self, linear: nn.Linear):
        self.weight, self.bias = linear.weight, linear.bias
        self.columns, self.depth = self.weight.shape
        self.packed: list[tuple[int, int, Tensor]] = []
        self.device = "ROCKCHIP"

    def prepare(self, domains: _Domains) -> None:
        if self.weight is None:
            return
        self.packed.clear()
        self.device = domains.place(round_up(self.columns, 32) * round_up(self.depth, 32) * 2)
        for c0 in range(0, self.columns, TILE):
            n = min(TILE, self.columns - c0)
            for k0 in range(0, self.depth, K_TILE):
                k = min(K_TILE, self.depth - k0)
                # realize per tile: a lazy graph would hold every tile's temporaries at once
                self.packed.append((c0, k0, pack_weight(self.weight[c0:c0 + n, k0:k0 + k].contiguous(), self.device).realize()))
        self.weight = None  # packed tiles carry everything; the source is the largest live copy

    def __call__(self, x: Tensor) -> Tensor:
        shape = x.shape
        dense = x.reshape(-1, shape[-1])
        rows = dense.shape[0]
        k_max = round_up(min(K_TILE, self.depth), 32)
        row_tile = max(1, min(64, 65536 // k_max))
        out_rows = []
        for r0 in range(0, rows, row_tile):
            part = dense[r0:r0 + row_tile]
            width = 24576 if part.shape[0] == 1 else 12288
            pieces = []
            for c0 in range(0, self.columns, TILE):
                n = min(TILE, self.columns - c0)
                for col in range(0, n, width):
                    w2 = min(width, n - col)
                    acc = None
                    for pc, pk, weight in self.packed:
                        if pc != c0:
                            continue
                        k = weight.shape[1] * 32
                        piece = packed_matmul(part[:, pk:pk + k], weight, w2, self.device, column_offset=col)
                        acc = piece if acc is None else acc + piece
                    if acc is None:
                        raise RuntimeError("NPULinear.prepare() was not called")
                    pieces.append(acc)
            out_rows.append(Tensor.cat(*pieces, dim=1) if len(pieces) > 1 else pieces[0])
        out = Tensor.cat(*out_rows, dim=0) if len(out_rows) > 1 else out_rows[0]
        out = out.reshape(*shape[:-1], self.columns)
        return out if self.bias is None else out + self.bias


def _walk(obj, visit) -> int:
    """Call visit on every Linear reachable from `obj`; returns the count."""
    count = 0

    def walk(node) -> None:
        nonlocal count
        for name, value in list(vars(node).items()):
            if isinstance(value, (nn.Linear, NPULinear)):
                new = visit(value)
                if new is not value:
                    setattr(node, name, new)
                count += 1
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, (nn.Linear, NPULinear)):
                        new = visit(item)
                        if new is not item:
                            value[index] = new
                        count += 1
                    elif hasattr(item, "__dict__"):
                        walk(item)
            elif hasattr(value, "__dict__") and not isinstance(value, (Tensor, dict, str, int, float)):
                walk(value)

    walk(obj)
    return count


def replace_linears(model) -> int:
    """Swap every nn.Linear reachable from `model` for NPULinear; returns the count."""
    return _walk(model, lambda linear: linear if isinstance(linear, NPULinear) else NPULinear(linear))


def prepare_linears(model) -> int:
    """Pack weights for every NPULinear reachable from `model`; returns the count."""
    def prepare(linear):
        if isinstance(linear, NPULinear):
            linear.prepare(_DOMAINS)
        return linear
    return _walk(model, prepare)
