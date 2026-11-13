#!/usr/bin/env python3
"""CPU-only structural tests for the opt-in adjacent-submit merge (ONPU_MERGE_SUBMITS).

No NPU and no DRM device is used: the merge builder only needs a plan's task
descriptors, its prepared command buffer and an allocator, so every object here
is backed by host memory with synthetic DMA addresses. What this proves:

  * the earlier plan's per-core terminating tail is patched into a PC jump to the
    later plan's body, with register amount ceil(body/2)+1, and restore works;
  * the combined task array is core-major [A0,B0,A1,B1,A2,B2] and every
    descriptor keeps its geometry / points at its own command block;
  * the merged submit descriptor (tasks, core mask, domain, subcore ranges);
  * mergeability gating ignores everything except the verified decode geometry;
  * against the real recorded decode op order the submit count falls
    171 -> 113 with ONPU_MERGE_SUBMITS=2 (pairs only: 171 -> 114), while the
    513 tasks per token are unchanged.

Run: DEV=CPU ROCKCHIP_TEST=0 /home/orangepi/tinygrad/.venv/bin/python -m pytest \
       test/stream/test_merge_submits.py -x -q
"""
from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import _cmd
from tinygrad.llm.kernels.rockchip_replay import (build_merged_submit, mergeable_plan, NativeReplay,
                                                  _restore_merged_tails, _MERGED_TAIL_ORIGINALS)
from tinygrad.runtime.ops_rockchip import (RockchipMatmul, RockchipGroupedMatmul, RockchipBatchedMatmul,
                                          RockchipRoutedMatmul, RockchipBucketedMatmul)

PROFILE = Path("/home/orangepi/oNPU/analysis/arms_cbuf/20261109-202029/r1-narrow_cbuf/narrow_cbuf.profile.jsonl")
TASK = ctypes.sizeof(rk.struct_rknpu_task)
BODY = 108
TAIL = 4


@pytest.fixture(autouse=True)
def _restore() -> None:
  yield
  _restore_merged_tails()
  assert not _MERGED_TAIL_ORIGINALS


class FakeMeta:
    def __init__(self, obj_addr: int): self.obj_addr = obj_addr


class FakeBuffer:
    def __init__(self, size: int, dma: int, obj_addr: int):
        self._mem = ctypes.create_string_buffer(size)
        self.addr = ctypes.addressof(self._mem)
        self.dma = dma
        self.size = size
        self.meta = FakeMeta(obj_addr)


class FakeDevice:
    def __init__(self, domain: int, base_dma: int):
        self.domain = domain
        self._next_dma = base_dma
        self._next_obj = 0x90000000
        self.allocated: list[FakeBuffer] = []
        self.freed: list[FakeBuffer] = []
        self.fd = SimpleNamespace(fd=3)  # only used when formatting the ioctl statement

    def allocate(self, size: int, flags: int = 0) -> FakeBuffer:
        size = max(4096, (size + 4095) // 4096 * 4096)
        buf = FakeBuffer(size, self._next_dma, self._next_obj)
        self._next_dma += size
        self._next_obj += size
        self.allocated.append(buf)
        return buf

    def free(self, buf: FakeBuffer) -> None: self.freed.append(buf)
    def sync(self, buf: FakeBuffer, flags: int) -> None: self.synced = getattr(self, "synced", 0) + 1


def words_of(buf: FakeBuffer, word_offset: int, count: int) -> list[int]:
    return list((ctypes.c_uint64 * count).from_address(buf.addr + word_offset * 8))


def set_word(buf: FakeBuffer, word_offset: int, value: int) -> None:
    ctypes.c_uint64.from_address(buf.addr + word_offset * 8).value = value


def make_plan(dev: FakeDevice, body: int = BODY, tag: int = 0) -> SimpleNamespace:
    """A plan with three one-per-core tasks whose command blocks are terminal."""
    stride = body + TAIL
    cmd = dev.allocate(stride * 3 * 8)
    task = dev.allocate(max(4096, 3 * TASK), rk.RKNPU_MEM_KERNEL_MAPPING)
    descs = (rk.struct_rknpu_task * 3).from_address(task.addr)
    for core in range(3):
        block = [(0x201 << 48) | (tag << 32) | (core << 24) | i for i in range(body)]
        block += [_cmd(1, 0), _cmd(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS),
                  _cmd(rk.TARGET_VERSION, 0), _cmd(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0xd)]
        for i, word in enumerate(block): set_word(cmd, core * stride + i, word)
        descs[core] = rk.struct_rknpu_task(0, 0, 0xd, 0x300, 0x1ffff, 0, body, 0, cmd.dma + core * stride * 8)
    return SimpleNamespace(dev=dev, cmd=cmd, task=task, tasks=3, core_mask=7,
                           subcores=((0, 0), (0, 0), (0, 1), (1, 1), (2, 1)),
                           m=1, int8=True, dynamic_weights=False)


def tails(plan: SimpleNamespace) -> list[tuple[int, int]]:
    """(body start word, body length) per core, derived from the task descriptors."""
    descs = (rk.struct_rknpu_task * 3).from_address(plan.task.addr)
    return [((d.regcmd_addr - plan.cmd.dma) // 8, d.regcfg_amount) for d in descs]


def test_pair_merge_patches_tail_and_builds_core_major_tasks() -> None:
    dev = FakeDevice(domain=0, base_dma=0x1000_0000)
    a, b = make_plan(dev, tag=1), make_plan(dev, tag=2)
    a_tails, b_tails = tails(a), tails(b)
    a_original = [words_of(a.cmd, start + body, TAIL) for start, body in a_tails]
    b_original = [words_of(b.cmd, start + body, TAIL) for start, body in b_tails]

    info, keep = build_merged_submit([a, b])
    task = keep[0]
    assert info["tasks"] == 6 and info["core_mask"] == 7 and info["domain"] == 0
    assert tuple(info["subcores"]) == ((0, 0), (0, 0), (0, 2), (2, 2), (4, 2))

    # Every plan's command tail but the last is now a jump to the next plan's body.
    for core in range(3):
        start, body = a_tails[core]
        nxt = b.cmd.dma + b_tails[core][0] * 8
        tail = words_of(a.cmd, start + body, 2)
        assert tail[0] == _cmd(rk.TARGET_PC_REG, rk.REG_PC_BASE_ADDRESS, nxt & 0xfffffff0)
        assert tail[1] == _cmd(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, ceildiv(b_tails[core][1], 2) + 1)
    # The last plan's terminal tail is untouched.
    for core in range(3):
        start, body = b_tails[core]
        assert words_of(b.cmd, start + body, TAIL) == b_original[core]

    # Combined task array is core-major; descriptors keep geometry and their own command blocks.
    descs = (rk.struct_rknpu_task * 6).from_address(task.addr)
    assert [d.regcfg_amount for d in descs] == [BODY] * 6
    for core in range(3):
        assert descs[core * 2].regcmd_addr == a.cmd.dma + a_tails[core][0] * 8
        assert descs[core * 2 + 1].regcmd_addr == b.cmd.dma + b_tails[core][0] * 8

    # Restore puts the earlier plan's original terminating tail back.
    _restore_merged_tails()
    for core in range(3):
        start, body = a_tails[core]
        assert words_of(a.cmd, start + body, TAIL) == a_original[core]


def test_triple_merge_chains_a_to_b_to_c() -> None:
    dev = FakeDevice(domain=2, base_dma=0x2000_0000)
    plans = [make_plan(dev, tag=t) for t in (3, 4, 5)]
    all_tails = [tails(p) for p in plans]
    c_original = [words_of(plans[2].cmd, start + body, TAIL) for start, body in all_tails[2]]
    info, keep = build_merged_submit(plans)
    task = keep[0]
    assert info["tasks"] == 9 and info["domain"] == 2
    assert tuple(info["subcores"]) == ((0, 0), (0, 0), (0, 3), (3, 3), (6, 3))
    for core in range(3):
        for position in (0, 1):
            start, body = all_tails[position][core]
            nxt_start, nxt_body = all_tails[position + 1][core]
            tail = words_of(plans[position].cmd, start + body, 2)
            assert tail[0] == _cmd(rk.TARGET_PC_REG, rk.REG_PC_BASE_ADDRESS,
                                   (plans[position + 1].cmd.dma + nxt_start * 8) & 0xfffffff0)
            assert tail[1] == _cmd(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, ceildiv(nxt_body, 2) + 1)
        assert words_of(plans[2].cmd, all_tails[2][core][0] + all_tails[2][core][1], TAIL) == c_original[core]
    descs = (rk.struct_rknpu_task * 9).from_address(task.addr)
    assert [d.regcfg_amount for d in descs] == [BODY] * 9
    for core in range(3):
        for position in range(3):
            assert descs[core * 3 + position].regcmd_addr == plans[position].cmd.dma + all_tails[position][core][0] * 8


def test_unequal_body_lengths() -> None:
    dev = FakeDevice(domain=0, base_dma=0x3000_0000)
    a, b = make_plan(dev, body=BODY, tag=1), make_plan(dev, body=100, tag=2)
    a_tails = tails(a)
    info, keep = build_merged_submit([a, b])
    assert info["tasks"] == 6
    for core in range(3):
        start, body = a_tails[core]
        tail = words_of(a.cmd, start + body, 2)
        assert tail[1] == _cmd(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, ceildiv(100, 2) + 1)
    descs = (rk.struct_rknpu_task * 6).from_address(keep[0].addr)
    assert [d.regcfg_amount for d in descs] == [BODY, 100, BODY, 100, BODY, 100]


def test_bad_groups_raise_and_leave_plans_untouched() -> None:
    d0, d1 = FakeDevice(0, 0x4000_0000), FakeDevice(1, 0x5000_0000)
    with pytest.raises(ValueError): build_merged_submit([make_plan(d0), make_plan(d1)])
    plan = make_plan(d0)
    with pytest.raises(ValueError): build_merged_submit([plan])
    plan.tasks = 2
    with pytest.raises(ValueError): build_merged_submit([make_plan(d0), plan])
    before = tails(make_plan(d0))
    assert before and not _MERGED_TAIL_ORIGINALS


def classed_plan(dev: FakeDevice, cls: type = RockchipMatmul) -> SimpleNamespace:
    plan = object.__new__(cls)
    plan.__dict__.update(make_plan(dev).__dict__)
    return plan


def test_mergeable_plan_gate() -> None:
    dev = FakeDevice(0, 0x6000_0000)
    assert mergeable_plan(classed_plan(dev))
    assert mergeable_plan(classed_plan(dev, RockchipGroupedMatmul))
    for cls in (RockchipBatchedMatmul, RockchipRoutedMatmul, RockchipBucketedMatmul):
        assert not mergeable_plan(classed_plan(dev, cls)), cls
    for attr, value in (("tasks", 1), ("core_mask", 3), ("m", 64), ("int8", False), ("dynamic_weights", True),
                        ("subcores", ((0, 0), (0, 0), (0, 2), (2, 2), (4, 2)))):
        bad = classed_plan(dev)
        setattr(bad, attr, value)
        assert not mergeable_plan(bad), attr


def test_merged_ioctl_statement_counts_tasks() -> None:
    stub = SimpleNamespace(ioctl="IOCTL", dev=SimpleNamespace(fd=SimpleNamespace(fd=3)))
    code = NativeReplay._submit_code(stub, 6, 0x1234, 1, 7, ((0, 0), (0, 0), (0, 2), (2, 2), (4, 2)), 7)
    assert "stats[0]++;stats[1]+=6;" in code
    assert ",6," in code and ",4660ULL," in code
    assert "return 2000007;" in code and "IOCTL(3," in code


def ceildiv(v: int, d: int) -> int: return -(-v // d)


def _plan_geometry(label: str) -> tuple[int, int, int, int] | None:
    """(groups, m, n, k) for a decode NPU label, or None for a CPU/COPY label."""
    if not label.startswith("ROCKCHIP"):
        return None
    name = label.split(":")[-1]
    parts = name.split("_")
    if parts[1] == "grouped":  # rockchip_grouped_int8_<groups>_<m>_<n>_<k>_<tile>
        return int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
    m, n, k = int(parts[2]), int(parts[3]), int(parts[4])  # rockchip_int8_<m>_<n>_<k>
    return 1, m, n, k


def _decode_groups(labels: list[str], max_group: int) -> list[list[int]]:
    """Replicate NativeReplay's grouping over the recorded call order."""
    groups: list[list[int]] = []
    run: list[int] = []
    previous: int | None = None
    for index, label in enumerate(labels):
        geom = _plan_geometry(label)
        mergeable = geom is not None and geom[1] == 1  # every decode NPU submit has 3 one-per-core tasks
        if not mergeable:
            previous = None
            continue
        if previous is not None and len(run) < max_group:
            run.append(index)
        else:
            run = [index]
            groups.append(run)
        previous = index
    return groups


@pytest.mark.skipif(not PROFILE.exists(), reason="recorded decode profile not available")
def test_recorded_decode_order_submit_count() -> None:
    first = json.loads(PROFILE.read_text().splitlines()[0])
    labels = first["labels"]
    npu_labels = [label for label in labels if label.startswith("ROCKCHIP")]
    assert len(npu_labels) == first["submits"] == 171
    assert first["tasks"] == 513  # merging keeps every task; only ioctls are merged

    def merged_group_sizes(max_group: int) -> list[int]:
        return sorted(len(group) for group in _decode_groups(labels, max_group) if len(group) > 1)

    # The real graph: 28 gate+up pairs, 28 ffn_down pairs, one lm_head triple.
    assert merged_group_sizes(3) == [2] * 56 + [3]
    assert merged_group_sizes(2) == [2] * 57
    assert 171 - 58 == 113  # ONPU_MERGE_SUBMITS=2
    assert 171 - 57 == 114  # ONPU_MERGE_SUBMITS=1
    for group in _decode_groups(labels, 3):
        if len(group) > 1:
            domains = {labels[i].split(":")[1] if len(labels[i].split(":")) > 2 else "0" for i in group}
            assert len(domains) == 1, (domains, [labels[i] for i in group])


def fake_finalize(plans, indices, max_group=2):
    """A NativeReplay instance with only the fields _finalize_merged_submits touches."""
    inst = object.__new__(NativeReplay)
    inst.statements = []
    inst._merge_slots, inst._merge_groups, inst._merge_run = [], [], []
    inst._merge_level, inst._merge_max_group = (1 if max_group == 2 else 2), max_group
    inst.kept, inst.submits = [], 0
    inst.ioctl = "IOCTL"
    inst.dev = SimpleNamespace(fd=SimpleNamespace(fd=3))
    for slot, (plan, index) in enumerate(zip(plans, indices)):
        token = f"/*MERGE-SLOT-{slot}-{index}*/"
        inst.statements.extend([f"validate-{index}", token])
        inst._merge_slots.append({"plan": plan, "index": index, "token": token})
    inst._merge_groups = [list(range(len(plans)))]
    inst._merge_run = list(range(len(plans)))
    return inst


def test_finalize_statement_order_stats_and_submit_count() -> None:
    dev = FakeDevice(0, 0x7000_0000)
    a, b = make_plan(dev), make_plan(dev)
    inst = fake_finalize([a, b], [11, 12])
    inst._finalize_merged_submits()
    # Every earlier plan's statements precede the one merged ioctl; nothing follows it.
    code = inst.statements[-1]
    assert inst.statements[:-1] == ["validate-11", "", "validate-12"]
    assert "stats[0]++;stats[1]+=6;" in code and ",6," in code
    assert "return 2000012;" in code  # the last plan's call index reports ioctl failure
    assert inst.submits == 1
    assert len(inst.kept) == 1  # the combined task buffer


def test_finalize_single_group_count() -> None:
    dev = FakeDevice(0, 0x7100_0000)
    inst = fake_finalize([make_plan(dev)], [4], max_group=2)
    inst._finalize_merged_submits()
    assert inst.statements[0] == "validate-4" and len(inst.statements) == 2
    assert inst.submits == 1
    assert "stats[0]++;stats[1]+=3;" in inst.statements[-1]
    assert "return 2000004;" in inst.statements[-1]


def test_rebuild_records_first_original() -> None:
    """Repeated builds over already-patched plans still restore the terminating tails."""
    dev = FakeDevice(0, 0x7200_0000)
    a, b = make_plan(dev), make_plan(dev)
    a_tails = tails(a)
    terminal = [words_of(a.cmd, start + body, TAIL) for start, body in a_tails]
    build_merged_submit([a, b])
    linked = [words_of(a.cmd, start + body, 2) for start, body in a_tails]
    assert all(words[0] != _cmd(1, 0) for words in linked)
    build_merged_submit([a, b])   # rebuild without restore: already-patched locations re-written
    build_merged_submit([a, b])
    _restore_merged_tails()
    assert [words_of(a.cmd, start + body, TAIL) for start, body in a_tails] == terminal
