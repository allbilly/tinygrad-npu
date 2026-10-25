"""CPU checks for calibration coverage, shared scales, and bounded weight reductions."""
from __future__ import annotations

import json, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.mali import MaliLinear
from extra.rockchip.calibrate_text_int8 import calibration_groups, collect_modules, load_scales, weight_column_max


class RecordingTensor:
  """Small proxy used to observe row-slice materialization without a device."""
  def __init__(self, tensor: Tensor):
    self.tensor = tensor
    self.shape = tensor.shape
    self.ndim = tensor.ndim
    self.dtype = tensor.dtype
    self.read_shapes: list[tuple[int, ...]] = []

  def __getitem__(self, index):
    part = self.tensor[index]
    owner = self

    class Slice:
      shape = part.shape

      def numpy(self):
        owner.read_shapes.append(tuple(part.shape))
        return part.numpy()

    return Slice()


def linear(depth: int, columns: int = 4) -> MaliLinear:
  weight = Tensor.zeros(columns, depth, device="CPU", dtype=dtypes.float)
  return MaliLinear(SimpleNamespace(weight=weight, bias=None), prepare_gemm=False)


def block(**layers) -> SimpleNamespace:
  return SimpleNamespace(**layers)


class CalibrationHelperTests(TestCase):
  def make_model(self):
    ordinary = block(
      attn_q=linear(8), attn_k=linear(8), attn_v=linear(8),
      ffn_gate=linear(8), ffn_up=linear(8), ffn_down=linear(16),
    )
    incomplete = block(attn_q=linear(8), attn_k=linear(8), ffn_gate=linear(8), ffn_up=linear(12))
    mismatched_qkv = block(attn_q=linear(8), attn_k=linear(8), attn_v=linear(16))
    gdn = block(
      attn_qkv=linear(8), attn_gate=linear(8), ssm_alpha=linear(8), ssm_beta=linear(8),
      ssm_g_a=linear(8), ssm_g_b=linear(8), ssm_f_a=linear(8), ssm_f_b=linear(8),
      ssm_out=linear(8), ffn_gate=linear(8), ffn_up=linear(8), ffn_down=linear(8),
    )
    return SimpleNamespace(blk=[ordinary, incomplete, mismatched_qkv, gdn], output=linear(8))

  def test_collects_nested_gdn_projection_names_and_skip_down_toggle(self):
    model = self.make_model()
    modules = collect_modules(model, skip_down=True)
    self.assertIn("blk.0.attn_q", modules)
    self.assertIn("blk.3.attn_qkv", modules)
    self.assertIn("blk.3.ssm_alpha", modules)
    self.assertIn("blk.3.ssm_g_a", modules)
    self.assertIn("blk.3.ssm_f_b", modules)
    self.assertIn("output", modules)
    self.assertNotIn("blk.0.ffn_down", modules)
    self.assertNotIn("blk.3.ffn_down", modules)

    with_down = collect_modules(model, skip_down=False)
    self.assertIn("blk.0.ffn_down", with_down)
    self.assertIn("blk.3.ffn_down", with_down)

  def test_collect_handles_cycles_and_shared_module_with_one_canonical_path(self):
    shared = linear(8)
    child = block(attn_qkv=shared)
    root = SimpleNamespace(blk=[child], alias=shared)
    child.parent = root
    found = collect_modules(root, skip_down=False)
    self.assertEqual(found["blk.0.attn_qkv"], shared)
    self.assertEqual(sum(module is shared for module in found.values()), 1)

  def test_dictionary_paths_and_wrapped_down_projection(self):
    down,other = linear(8),linear(8)
    model = SimpleNamespace(extra={'branch':[block(ffn_down=block(source=down))],'projection':other})
    self.assertEqual(collect_modules(model),{'extra.projection':other})
    self.assertEqual(collect_modules(model,skip_down=False),
                     {'extra.branch.0.ffn_down.source':down,'extra.projection':other})

  def test_groups_only_complete_same_parent_and_depth_matched_pairs(self):
    modules = collect_modules(self.make_model(), skip_down=False)
    groups = calibration_groups(modules)
    values = [set(members) for members in groups.values()]
    qkv = {"blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v"}
    gate_up = {"blk.0.ffn_gate", "blk.0.ffn_up"}
    gdn_gate_up = {"blk.3.ffn_gate", "blk.3.ffn_up"}
    self.assertIn(qkv, values)
    self.assertIn(gate_up, values)
    self.assertIn(gdn_gate_up, values)
    for name in ("blk.1.attn_q", "blk.1.attn_k", "blk.1.ffn_gate", "blk.1.ffn_up",
                 "blk.2.attn_q", "blk.2.attn_k", "blk.2.attn_v", "blk.3.attn_qkv"):
      self.assertIn({name}, values)
    flattened = [name for members in groups.values() for name in members]
    self.assertEqual(sorted(flattened), sorted(modules))

  def test_load_scales_roundtrip_preserves_shared_pair_identity(self):
    group = {
      "members": ["blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v"],
      "scales": [0.25] * 8,
    }
    payload = {"schema_version": 1, "checkpoint_sha256": "a" * 64, "groups": {"blk.0.attn_qkv": group}}
    with self.subTest("temporary CPU calibration file"):
      with tempfile.TemporaryDirectory(prefix="calibration-scales-") as directory:
        path = Path(directory) / "scales.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        scales = load_scales(path, "a" * 64)
        self.assertIs(scales["blk.0.attn_q"], scales["blk.0.attn_k"])
        self.assertIs(scales["blk.0.attn_k"], scales["blk.0.attn_v"])

  def test_weight_column_max_streams_multiple_small_chunks_and_matches_reference(self):
    rows, columns = 67, 64
    matrix = np.random.default_rng(712).normal(size=(rows, columns)).astype(np.float32)
    weight = RecordingTensor(Tensor(matrix, device="CPU", dtype=dtypes.float))
    actual = weight_column_max(weight, max_chunk_bytes=columns * 4 * 7)
    expected = np.abs(matrix).max(axis=0).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected)
    self.assertGreater(len(weight.read_shapes), 1)
    self.assertTrue(all(0 < shape[0] <= 7 and shape[0] < rows for shape in weight.read_shapes))

  def test_weight_max_preserves_fp16_extremes_and_rejects_invalid_values(self):
    matrix = np.array([[0.,-65504.,2**-24],[0.,65504.,-2**-24]],dtype=np.float16)
    np.testing.assert_array_equal(weight_column_max(Tensor(matrix,device='CPU'),max_chunk_bytes=1),
                                  np.abs(matrix.astype(np.float32)).max(axis=0))
    for values in ([[float('nan')]],[[float('inf')]],[[float('-inf')]]):
      with self.subTest(values=values),self.assertRaisesRegex(ValueError,'finite'):
        weight_column_max(Tensor(values,device='CPU'))
    for budget in (0,-1,True):
      with self.subTest(budget=budget),self.assertRaisesRegex(ValueError,'budget'):
        weight_column_max(Tensor([[1.]],device='CPU'),max_chunk_bytes=budget)
