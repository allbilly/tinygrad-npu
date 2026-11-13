"""CPU checks for calibration coverage, shared scales, and bounded weight reductions."""
from __future__ import annotations

import json, tempfile
import os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.mali import MaliLinear
from extra.rockchip.calibrate_text_int8 import calibration_groups, capture_activations, collect_modules, load_scales, weight_column_max


class RecordingTensor:
  """Small proxy used to observe row-slice materialization without a device."""
  def __init__(self, tensor: Tensor):
    self.tensor = tensor
    self.shape = tensor.shape
    self.ndim = tensor.ndim
    self.dtype = tensor.dtype
    self.device = tensor.device
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
  def test_cpu_reference_gemm_bounds_weight_reads_and_restores_on_failure(self):
    from tinygrad import Device, nn
    from extra.rockchip.calibrate_text_int8 import cpu_reference_execution
    original=nn.Linear.__call__
    original_embedding=nn.Embedding.__call__
    previous_lru=Device['CPU'].allocator.lru
    for dtype,npdtype in ((dtypes.half,np.float16),(dtypes.float,np.float32)):
      values=(np.arange(48).reshape(2,3,8)/8).astype(npdtype)
      weights=((np.arange(40).reshape(5,8)-20)/8).astype(npdtype)
      layer=nn.Linear(8,5,bias=True)
      layer.weight=RecordingTensor(Tensor(weights,device='CPU'))
      layer.bias=Tensor(np.arange(5,dtype=npdtype)/4,device='CPU')
      expected=(values.astype(np.float32)@weights.astype(np.float32).T).astype(npdtype)+layer.bias.numpy()
      with cpu_reference_execution(nn.Linear,max_chunk_bytes=2*8*(np.dtype(npdtype).itemsize+4)):
        self.assertFalse(Device['CPU'].allocator.lru)
        actual=layer(Tensor(values,device='CPU')).numpy()
      self.assertIs(nn.Linear.__call__,original)
      self.assertEqual(Device['CPU'].allocator.lru,previous_lru)
      np.testing.assert_array_equal(actual.view(np.uint16 if npdtype is np.float16 else np.uint32),
                                    expected.view(np.uint16 if npdtype is np.float16 else np.uint32))
      self.assertEqual(layer.weight.read_shapes,[(2,8),(2,8),(1,8)])
      with self.assertRaisesRegex(RuntimeError,'reference failure'), \
           patch('numpy.matmul',side_effect=RuntimeError('reference failure')),cpu_reference_execution(nn.Linear):
        layer(Tensor(values,device='CPU'))
      self.assertIs(nn.Linear.__call__,original)
      self.assertIs(nn.Embedding.__call__,original_embedding)
      self.assertEqual(Device['CPU'].allocator.lru,previous_lru)

  def test_cpu_reference_embedding_preserves_rows_invalid_indices_and_zero_bits(self):
    from tinygrad import nn
    from extra.rockchip.calibrate_text_int8 import cpu_reference_execution
    original=nn.Embedding.__call__
    for npdtype in (np.float16,np.float32):
      table=(np.arange(48).reshape(6,8)/8).astype(npdtype)
      table[1,0]=-0.0
      layer=nn.Embedding(6,8)
      layer.weight=Tensor(table,device='CPU')
      indices=Tensor([[1,4,1,-1,6]],device='CPU',dtype=dtypes.int32)
      expected=layer(indices).numpy()
      with cpu_reference_execution(nn.Linear):actual=layer(indices).numpy()
      self.assertIs(nn.Embedding.__call__,original)
      bits=np.uint16 if npdtype is np.float16 else np.uint32
      np.testing.assert_array_equal(actual.view(bits),expected.view(bits))

  def test_cpu_reference_cli_observes_checkpoint_layers_without_acceleration(self):
    from tinygrad import Device, nn
    from tinygrad.llm.gguf import gguf_shard
    from extra.rockchip.calibrate_text_int8 import main
    class Model:
      def __init__(self):
        self.blk=[]
        self.config=SimpleNamespace(num_experts=0)
        self.output=nn.Linear(8,4,bias=False)
        self.output.weight=Tensor.ones(4,8,device='CPU',dtype=dtypes.half)
      def logits(self,ids,pos):return self.output(ids.half().unsqueeze(-1).expand(1,ids.shape[1],8))
    before=set(Device._opened_devices)
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      checkpoint=root/'weights.gguf'
      checkpoint.write_bytes(np.ones((4,8),np.float16).tobytes())
      model=Model()
      model.output.weight=gguf_shard({'weight':(Tensor(checkpoint),(4,8),1)},('CPU',),mapped_cpu=True)['weight']
      data=root/'prompts.json'
      data.write_text(json.dumps([{'input':'separate calibration text'}]))
      out=root/'scales.json'
      argv=['calibrate_text_int8','--checkpoint',str(checkpoint),'--dataset',str(data),'--output',str(out),
            '--device','CPU','--cpu-reference','--no-skip-down']
      with patch.dict(os.environ,{},clear=False),patch.object(sys,'argv',argv), \
           patch('tinygrad.llm.model.Transformer.from_gguf',return_value=(model,{})), \
           patch('tinygrad.llm.cli.SimpleTokenizer.from_gguf_kv',return_value=SimpleNamespace(encode=lambda _: [1,2])), \
           patch('tinygrad.llm.kernels.rockchip.configure',side_effect=AssertionError('NPU setup is forbidden')), \
           patch('tinygrad.llm.kernels.mali.configure',side_effect=AssertionError('GPU setup is forbidden')):
        main()
      record=json.loads(out.read_text())
      self.assertTrue(record['cpu_reference'])
      self.assertIsNone(record['cpu_npu_placement'])
      self.assertIsNone(record['gpu_kernels'])
      self.assertEqual(record['npu_submissions'],0)
      self.assertEqual(record['activation_statistics_dtype'],'float32')
      self.assertTrue(all(name.split(':')[0] in ('CPU','NPY','PYTHON','DISK') for name in record['opened_devices']))
      self.assertEqual(record['observed_projections'],['output'])
      # Half inputs must widen before exponentiation of statistics, avoiding
      # half-rounded sqrt(2) scales in the reference calibration.
      actual=np.asarray(record['groups']['output']['scales'],np.float32)
      expected=np.full(8,np.sqrt(np.float32(2)),np.float32)
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    self.assertTrue(all(name.split(':')[0] in ('CPU','NPY','PYTHON','DISK') for name in set(Device._opened_devices)-before))

  def test_cpu_reference_rejects_gpu_selection_before_loading(self):
    from extra.rockchip.calibrate_text_int8 import main
    argv=['calibrate_text_int8','--checkpoint','unused.gguf','--dataset','unused.json','--output','unused-output.json',
          '--device','CL','--cpu-reference']
    with patch.object(sys,'argv',argv),patch('tinygrad.llm.model.Transformer.from_gguf') as load:
      with self.assertRaises(SystemExit) as error:main()
      self.assertEqual(error.exception.code,2)
      load.assert_not_called()

  def test_cpu_cli_startup_uses_bounded_unfused_placement_and_restores_allocator(self):
    from tinygrad import Device, nn
    from tinygrad.llm.kernels.rockchip import RockchipLinear
    from extra.rockchip.calibrate_text_int8 import main
    source=nn.Linear(8,4,bias=False)
    source.weight=Tensor.ones(4,8,device='CPU')
    class Model:
      def __init__(self):
        self.blk=[]
        self.config=SimpleNamespace(num_experts=0)
        self.output=RockchipLinear(source,'off','off')
      def logits(self,ids,pos):return self.output(ids.float().unsqueeze(-1).expand(1,ids.shape[1],8))
    allocator=Device['CPU'].allocator
    previous=allocator.lru
    before=set(Device._opened_devices)
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      checkpoint=root/'weights.gguf'
      checkpoint.write_bytes(b'cpu-fixture')
      data=root/'prompts.json'
      data.write_text(json.dumps([{'input':'separate calibration text'}]))
      for fail in (False,True):
        out=root/('failed.json' if fail else 'scales.json')
        argv=['calibrate_text_int8','--checkpoint',str(checkpoint),'--dataset',str(data),'--output',str(out),'--device','CPU']
        with patch.dict(os.environ,{},clear=False),patch.object(sys,'argv',argv), \
             patch('tinygrad.llm.model.Transformer.from_gguf',return_value=(Model(),{})) as load, \
             patch('tinygrad.llm.cli.SimpleTokenizer.from_gguf_kv',return_value=SimpleNamespace(encode=lambda _: [1,2])), \
             patch('tinygrad.llm.kernels.rockchip.configure',side_effect=RuntimeError('prepare failure') if fail else None,
                   return_value={'precision':'int8'}) as configure:
          if fail:
            with self.assertRaisesRegex(RuntimeError,'prepare failure'):main()
            self.assertFalse(out.exists())
          else:
            main()
            record=json.loads(out.read_text())
            self.assertEqual(record['observed_projections'],['output'])
            self.assertIn('INT8 CPU/NPU',record['capture_mode'])
            self.assertIsNone(record['gpu_kernels'])
            self.assertEqual(record['npu_submissions'],0)
          self.assertEqual(load.call_args.kwargs,{'max_context':512,'realize':False,'mapped_cpu':True})
          self.assertTrue(configure.call_args.kwargs['cpu_attention'])
          self.assertEqual(configure.call_args.kwargs['precision'],'int8')
          self.assertFalse(configure.call_args.kwargs.get('fuse_projections',False))
        self.assertEqual(allocator.lru,previous)
    self.assertFalse(any(d.startswith(('CL','ROCKCHIP')) for d in set(Device._opened_devices)-before))

  def test_collects_cpu_npu_wrappers_without_preparing_or_opening_the_npu(self):
    from tinygrad import Device
    from tinygrad.llm.kernels.rockchip import RockchipLinear
    from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
    before=set(Device._opened_devices)
    def wrapped():return RockchipInt8Linear(SimpleNamespace(weight=Tensor.zeros(4,8,device='CPU'),bias=None),'on','on')
    q,k,v,gate,up,down=(wrapped() for _ in range(6))
    model=SimpleNamespace(blk=[block(attn_q=q,attn_k=k,attn_v=v,ffn_gate=gate,ffn_up=up,ffn_down=down)])
    found=collect_modules(model,linear_type=RockchipLinear)
    self.assertEqual(set(found),{'blk.0.attn_q','blk.0.attn_k','blk.0.attn_v','blk.0.ffn_gate','blk.0.ffn_up'})
    self.assertEqual(set(map(tuple,calibration_groups(found).values())),
                     {('blk.0.attn_q','blk.0.attn_k','blk.0.attn_v'),('blk.0.ffn_gate','blk.0.ffn_up')})
    self.assertTrue(all(not layer.packed for layer in found.values()))
    self.assertFalse(any(d.startswith(('CL','ROCKCHIP')) for d in set(Device._opened_devices)-before))

  def test_capture_observes_cpu_inputs_and_restores_hooks_after_success_and_failure(self):
    import tinygrad.llm.model as model_module
    class Projection:
      def __call__(self,x):return x+1
    class Model:
      def __init__(self,mode):self.projection,self.mode=Projection(),mode
      def logits(self,ids,pos):
        x=ids.float().unsqueeze(-1)
        out=self.projection(x.cat(-x,dim=-1))
        if self.mode=='failure':raise RuntimeError('forward failure')
        return out if self.mode!='nonfinite' else out*float('nan')
    tokenizer=SimpleNamespace(encode=lambda text:list(map(int,text.split())))
    original,original_function=Projection.__call__,model_module.function
    for mode in ('success','failure','nonfinite','empty','missing'):
      with self.subTest(mode=mode):
        model=Model(mode)
        modules={'projection':model.projection}
        if mode=='missing':modules['unused']=Projection()
        data=[{'input':'1 2'},{'input':'7'}] if mode!='empty' else [{'input':''}]
        if mode=='success':
          actual=capture_activations(model,modules,Projection,data,tokenizer,'CPU')
          np.testing.assert_array_equal(actual['projection'],[7,7])
        else:
          with self.assertRaises((RuntimeError,ValueError)):capture_activations(model,modules,Projection,data,tokenizer,'CPU')
        self.assertIs(Projection.__call__,original)
        self.assertIs(model_module.function,original_function)

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
