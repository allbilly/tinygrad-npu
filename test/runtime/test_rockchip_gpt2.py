import os, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, Variable, nn
from tinygrad.llm.kernels.rockchip_gpt2 import GPT2Head, PrefillJit, configure_gpt2, half_attention, greedy_argmax
from extra.rockchip.monitor_npu import parse_load


class TestGPT2Adapter(unittest.TestCase):
  def test_greedy_argmax_constant_logits(self):
    self.assertEqual(greedy_argmax(Tensor.ones(1,193,device='CPU')).tolist(),[0])

  def test_greedy_argmax_ties_tail_and_signed_zero(self):
    rng = np.random.default_rng(22)
    for columns in (1,3,16,17,193,50257):
      values = rng.normal(size=(2,columns)).astype(np.float32)
      values[0] = -1
      values[0,0],values[0,-1] = -0.,0.
      values[1,columns//2],values[1,-1] = 100.,100.
      actual = greedy_argmax(Tensor(values,device='CPU')).numpy()
      np.testing.assert_array_equal(actual,values.argmax(axis=-1))
  def test_head_only_projects_final_token(self):
    layer = nn.Linear(32,96,bias=False)
    layer.weight = Tensor(np.arange(96*32,dtype=np.float32).reshape(96,32)/1000,device='CPU')
    head = GPT2Head(layer,'off','off')
    x = Tensor(np.arange(5*32,dtype=np.float32).reshape(1,5,32)/100,device='CPU')
    np.testing.assert_array_equal(head(x).numpy(),layer(x[:,-1:]).numpy())
    self.assertEqual(head(x).shape,(1,1,96))

  def test_prefill_shapes_are_bounded_and_keep_phase(self):
    layer = SimpleNamespace(phase=None)
    phases = []
    def forward(x,start,temp):
      phases.append(layer.phase)
      return (x*2+1).realize()
    run = PrefillJit(forward,[layer],False)
    for rows in [4,8,4,8,4,8,*range(9,19),18,18]:
      x = Tensor(np.arange(rows,dtype=np.float32).reshape(1,rows),device='CPU').realize()
      np.testing.assert_array_equal(run(x,Variable('pos',0,63).bind(0)).numpy(),x.numpy()*2+1)
      self.assertIsNone(layer.phase)
      self.assertLessEqual(len(run.variants),8)
    self.assertEqual(set(phases),{'prefill'})

  def test_load_requires_real_per_core_percentages(self):
    self.assertEqual(parse_load('NPU load: Core0: 0%, Core1: 17%, Core2: 100%,'),[0,17,100])
    for value in ('100@1000000000Hz','Core0: 20%','Core0: 101%, Core1: 0%, Core2: 0%'):
      with self.assertRaises(ValueError): parse_load(value)

  def test_half_attention_full_outputs_and_inactive_cache(self):
    rng = np.random.default_rng(19)
    for rows,start in [(1,0),(1,31),(8,0),(17,31),(17,111)]:
      data = rng.normal(0,.3,(2,1,4,128,64)).astype(np.float16)
      data[:,:,:,start+rows:] = np.nan
      cache = Tensor(data,device='CPU')
      query = Tensor(rng.normal(0,.3,(1,4,rows,64)).astype(np.float16),device='CPU')
      mask = Tensor.full((1,1,rows,start+rows),float('-inf'),device='CPU',dtype=query.dtype).triu(start+1)
      expected = query.scaled_dot_product_attention(cache[0][:,:,:start+rows],cache[1][:,:,:start+rows],mask).numpy()
      actual = half_attention(query,cache,start).numpy()
      self.assertTrue(np.isfinite(actual).all())
      np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=3e-3)

  def test_softmax_denominator_keeps_half_exponent_rounding(self):
    data = np.zeros((2,1,1,128,4),dtype=np.float16)
    data[0,0,0,:,0] = np.random.default_rng(3).normal(0,3,128).astype(np.float16)
    data[1] = np.array([1,2,4,8],dtype=np.float16)
    query = Tensor(np.array([[[[2,0,0,0]]]],dtype=np.float16),device='CPU')
    cache = Tensor(data,device='CPU')
    expected = query.scaled_dot_product_attention(cache[0],cache[1]).numpy()
    np.testing.assert_array_equal(half_attention(query,cache,127).numpy(),expected)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_warmed_cpu_model_switches_to_native_npu_and_keeps_tokens(self):
    import examples.gpt2 as gpt2
    from tinygrad.nn.state import get_state_dict
    with patch.object(gpt2,'HALF',True),patch.object(gpt2,'MAX_CONTEXT',64):
      Tensor.manual_seed(24)
      model = gpt2.Transformer(dim=128,n_heads=4,n_layers=1,norm_eps=1e-5,vocab_size=193,max_seq_len=64)
      for tensor in get_state_dict(model).values(): tensor.replace(tensor.half().realize())
      def request(bulk=False, steps=7):
        if hasattr(model.h[0].attn,'cache_kv'):
          model.h[0].attn.cache_kv.assign(Tensor.zeros_like(model.h[0].attn.cache_kv)).realize()
        result = model(Tensor([[1,2,3,4]],device='CPU').realize(),Variable('start_pos',0,63).bind(0),0.0).tolist()
        if bulk:
          result.extend(model.rockchip_decode_many(result[-1],4,steps))
          return result
        for position in range(4,4+steps):
          result.extend(model(Variable('tokens',0,192).bind(result[-1]),Variable('start_pos',1,63).bind(position),0.0).tolist())
        return result
      expected = request()
      expected_long = request(steps=40)
      for _ in range(3): self.assertEqual(request(),expected)
      configure_gpt2(model)
      affinity = os.sched_getaffinity(0)
      before = Device['ROCKCHIP'].submit_count
      for _ in range(6): self.assertEqual(request(),expected)
      before_bulk = Device['ROCKCHIP'].submit_count
      for _ in range(3): self.assertEqual(request(True),expected)
      self.assertGreater(model.rockchip_gpt2['loop'].tokens,0)
      self.assertEqual(Device['ROCKCHIP'].submit_count-before_bulk,3*8*5)
      self.assertEqual(request(True,40),expected_long)
      with self.assertRaises(ValueError): model.rockchip_decode_many(0,63,2)
      with self.assertRaises(ValueError): model.rockchip_decode_many(193,4,1)
      self.assertEqual(request(),expected)
      self.assertGreater(Device['ROCKCHIP'].submit_count,before)
      self.assertGreater(model.rockchip_gpt2['decode'].captured.native_runs,0)
      self.assertGreater(model.rockchip_gpt2['decode'].captured.native.clean_sites,0)
      self.assertGreater(model.rockchip_gpt2['decode'].captured.native.driver_from_sites,0)
      self.assertTrue(all(jit.captured.native_runs>0 for jit in model.rockchip_gpt2['prefill'].variants.values()))
      self.assertEqual(os.sched_getaffinity(0),affinity)
      helper = model.rockchip_gpt2['decode'].helper
      self.assertIsNotNone(helper)
      self.assertGreater(helper.runs,0)
      tid = helper.lib.spinner_tid()
      self.assertEqual(os.sched_getaffinity(tid),{0})
      self.assertEqual(os.getpriority(os.PRIO_PROCESS,tid),19)
      def ticks():
        row = Path(f'/proc/self/task/{tid}/stat').read_text().rsplit(')',1)[1].split()
        return int(row[11])+int(row[12])
      before = ticks()
      import time
      time.sleep(.15)
      self.assertEqual(ticks(),before)
      Device['ROCKCHIP'].reset()
      for _ in range(3): self.assertEqual(request(),expected)
      from tinygrad.llm.kernels.rockchip_pool import attention_pool
      pool = attention_pool()
      previous_jobs = pool.lib.pool_calls()
      model.logits(Variable('tokens',0,192).bind(1),Variable('start_pos',1,63).bind(20)).realize()
      self.assertEqual(pool.lib.pool_calls(),previous_jobs)
      self.assertFalse(pool.active)
      for position in (14,15,31,127):
        rng = np.random.default_rng(position)
        q = Tensor(rng.normal(0,.3,(1,12,1,64)).astype(np.float16),device='CPU')
        cache = Tensor(rng.normal(0,.3,(2,1,12,128,64)).astype(np.float16),device='CPU')
        serial = half_attention(q,cache,position).numpy()
        with pool: parallel = half_attention(q,cache,position,pool=pool.address).numpy()
        np.testing.assert_array_equal(parallel,serial)
      self.assertGreater(pool.lib.pool_calls(),0)
      import time
      time.sleep(.05)
      for i in range(3):
        worker_tid = pool.lib.pool_thread_tid(i)
        self.assertEqual(os.sched_getaffinity(worker_tid),{5+i})
        state = Path(f'/proc/self/task/{worker_tid}/stat').read_text().rsplit(')',1)[1].split()[0]
        self.assertEqual(state,'S')
      # Rebinding and complete FP32 outputs exercise real user-space TO_DEVICE
      # cleaning while every NPU-to-CPU invalidation remains in the driver.
      from tinygrad import TinyJit
      from tinygrad.llm.kernels.rockchip import pack_weight, packed_matmul
      from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
      rng = np.random.default_rng(23)
      for rows,depth,columns in [(1,768,2304),(21,3072,768),(64,768,3072)]:
        weights = rng.normal(0,.1,(columns,depth)).astype(np.float16)
        packed = pack_weight(Tensor(weights,device='CPU')).realize()
        fn = TinyJit(lambda x:packed_matmul(x,packed,columns).realize())
        for i in range(7):
          data = rng.normal(0,.1,(rows,depth)).astype(np.float16)
          actual = fn(Tensor(data,device='CPU').realize()).numpy()
          np.testing.assert_allclose(actual,data.astype(np.float32)@weights.astype(np.float32).T,atol=2e-4,rtol=3e-4)
          if i==1: fn.captured = RockchipCapturedJit(fn.captured,model.rockchip_gpt2['decode'].cleaner)
        self.assertGreater(fn.captured.native_runs,0)
        self.assertGreater(fn.captured.native.clean_sites,0)
        self.assertGreater(fn.captured.native.driver_from_sites,0)


if __name__=='__main__': unittest.main()
