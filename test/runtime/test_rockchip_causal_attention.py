import ctypes, functools, os, subprocess, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad.llm.kernels.rockchip_cpu import _fast_softmax_source

ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'tinygrad/llm/kernels/rockchip_causal_attention.c'

@functools.lru_cache(None)
def compile_helper(context:int,shared:bool=False,row_major:bool=False,token_major:bool=False):
  directory=tempfile.TemporaryDirectory(prefix='rockchip-causal-attention-cpu-')
  source=Path(directory.name)/'helper.c'
  library=Path(directory.name)/'helper.so'
  source.write_text(_fast_softmax_source()+'\n#define ATTENTION_HELPER_API\n#define SHARE_KV '+str(int(shared))+
                   '\n#define ROW_MAJOR '+str(int(row_major))+'\n#define TOKEN_MAJOR_OUTPUT '+str(int(token_major))+
                   '\n#define CONTEXT '+str(context)+'\n'+HELPER.read_text()+
                   '\nint test_bad_half4(const ushort4 *bits){return bad_half4(__builtin_bit_cast(half4,*bits));}\n')
  subprocess.run(['clang','-O2','-shared','-fPIC','-mcpu=native','-ffp-contract=off','-fno-math-errno',
                  str(source),'-lm','-o',str(library)],check=True,capture_output=True,text=True)
  lib=ctypes.CDLL(str(library))
  ptr=ctypes.c_void_p
  lib.pack.argtypes=[ctypes.c_int]*4+[ptr]*5
  lib.pack.restype=ctypes.c_int
  lib.normalize.argtypes=[ctypes.c_int]*4+[ptr]*2
  lib.normalize.restype=ctypes.c_int
  lib.unpack_result.argtypes=[ctypes.c_int]*3+[ptr]*2
  lib.unpack_result.restype=ctypes.c_int
  lib.set_pool.argtypes=[ctypes.c_ulonglong]
  lib.set_pool.restype=None
  lib.test_bad_half4.argtypes=[ptr]
  lib.test_bad_half4.restype=ctypes.c_int
  lib._test_tempdir=directory
  return lib

def ptr(array): return ctypes.c_void_p(array.ctypes.data)

def pack_oracle(query,cache,capacity,active,total,padded,context,shared=False,row_major=False):
  logical=np.zeros((16,capacity,128),np.float16)
  logical[:,:active]=query[:,:active].astype(np.float16)
  qa=np.concatenate([logical[:,r:r+64].reshape(16,-1,16,8).transpose(0,2,1,3).reshape(16,-1)
                     for r in range(0,capacity,64)],axis=1)
  if row_major:qa=logical
  key,value=(np.zeros((16,padded,128),np.float16) for _ in range(2))
  key[:,:total]=np.repeat(cache[0,0,:,:total],2,axis=0)
  value[:,:total]=np.repeat(cache[1,0,:,:total],2,axis=0)
  ka=key.reshape(16,padded//16,16,4,32).transpose(0,1,3,2,4).copy()
  va=value.transpose(0,2,1).reshape(16,8,16,padded//32,32).transpose(0,1,3,2,4).copy()
  qa,ka,va=qa.reshape(16,capacity,128),ka.reshape(16,padded,128),va.reshape(16,padded,128)
  return (qa[list(range(0,16,2))+list(range(1,16,2))],ka[::2],va[::2]) if shared else (qa,ka,va)

def score_layout(logical,capacity,padded,shared=False,row_major=False):
  if shared:logical=logical[list(range(0,16,2))+list(range(1,16,2))]
  if row_major:return logical.copy()
  return np.concatenate([logical[:,r:r+64].reshape(16,-1,padded//4,4).transpose(0,2,1,3).reshape(16,-1)
                         for r in range(0,capacity,64)],axis=1).reshape(16,capacity,padded).copy()

def probability_oracle(scores,capacity,active,total,padded,shared=False,row_major=False):
  prob=np.zeros((16,capacity,padded),np.float16)
  for r in range(active):
    used=total-active+r+1
    values=scores[:,r,:used].astype(np.float32)*np.float32(1/np.sqrt(128))
    ex=np.exp(values-values.max(-1,keepdims=True))
    prob[:,r,:used]=(ex/ex.sum(-1,keepdims=True)).astype(np.float16)
  if shared:prob=prob[list(range(0,16,2))+list(range(1,16,2))]
  if row_major:return prob
  return np.concatenate([prob[:,r:r+64].reshape(16,-1,padded//8,8).transpose(0,2,1,3).reshape(16,-1)
                         for r in range(0,capacity,64)],axis=1).reshape(16,capacity,padded)

class TestRockchipCausalAttentionCPU(unittest.TestCase):
  def test_half_guard_all_encodings_and_individual_lanes(self):
    lib=compile_helper(64)
    for first in range(0,65536,4):
      bits=np.arange(first,first+4,dtype=np.uint16)
      self.assertEqual(bool(lib.test_bad_half4(ptr(bits))),bool(np.any((bits&0x7c00)==0x7c00)))
    for lane in range(4):
      for invalid in (0x7c00,0xfc00,0x7c01,0xffff):
        bits=np.full(4,0x3c00,dtype=np.uint16)
        bits[lane]=invalid
        self.assertTrue(lib.test_bad_half4(ptr(bits)))

  def test_serial_and_parked_workers_preserve_exact_payloads(self):
    from tinygrad.llm.kernels.rockchip_pool import attention_pool
    pool=attention_pool()
    lib=compile_helper(256,True,True,True)
    rng=np.random.default_rng(20261030)
    capacity,active,total,padded=65,17,190,256
    query=rng.normal(0,.2,(16,capacity,128)).astype(np.float32)
    query[:,active:]=np.nan
    cache=rng.normal(0,.2,(2,1,8,256,128)).astype(np.float16)
    cache[:,:,:,total:]=np.nan
    results=[]
    affinity=os.sched_getaffinity(0)
    try:
      for address in (0,pool.prefill_address):
        qa=np.empty((16,capacity,128),np.float16)
        ka=np.empty((8,padded,128),np.float16)
        va=np.empty_like(ka)
        with pool.prefill():
          lib.set_pool(address)
          self.assertEqual(lib.pack(capacity,active,total,padded,ptr(query),ptr(cache),ptr(qa),ptr(ka),ptr(va)),0)
        results.append((qa,ka,va))
        self.assertEqual(os.sched_getaffinity(0),affinity)
      for a,b in zip(*results):np.testing.assert_array_equal(a.view(np.uint16),b.view(np.uint16))
    finally:lib.set_pool(0)

  def test_nonfinite_visible_score_is_rejected(self):
    lib=compile_helper(64)
    scores=np.zeros((16,17,64),np.float32)
    probabilities=np.empty((16,17,64),np.float16)
    for invalid in (np.nan,np.inf,-np.inf):
      scores.flat[0]=invalid
      self.assertEqual(lib.normalize(17,17,62,64,ptr(scores),ptr(probabilities)),11)

  def test_pack_normalize_unpack_dynamic_capacity_and_poison(self):
    cases=((128,17,62,64),(17,1,128,128),(65,17,190,256),(128,128,512,512),(65,1,62,64))
    formats=((False,False,False),(True,False,False),(True,True,True))
    for (shared,row_major,token_major),(capacity,active,total,padded) in ((fmt,case) for fmt in formats for case in cases):
      context=max(padded,64)
      lib=compile_helper(context,shared,row_major,token_major)
      lib.set_pool(0)
      rng=np.random.default_rng(20261028+capacity+active+total)
      q=rng.normal(0,.2,(16,capacity,128)).astype(np.float32)
      q[:,active:]=np.nan
      cache=rng.normal(0,.2,(2,1,8,context,128)).astype(np.float16)
      cache[:,:,:,total:]=np.nan
      qa=np.full((16,capacity,128),np.nan,np.float16)
      ka=np.empty((8 if shared else 16,padded,128),np.float16)
      va=np.empty_like(ka)
      code=lib.pack(capacity,active,total,padded,ptr(q),ptr(cache),ptr(qa),ptr(ka),ptr(va))
      self.assertEqual(code,0)
      expected=pack_oracle(q,cache,capacity,active,total,padded,context,shared,row_major)
      for actual,wanted in zip((qa,ka,va),expected):np.testing.assert_array_equal(actual.view(np.uint16),wanted.view(np.uint16))

      logical=np.full((16,capacity,padded),np.nan,np.float32)
      for h in range(16):
        for r in range(active):
          used=total-active+r+1
          logical[h,r,:used]=rng.normal(0,.4,used).astype(np.float32)
      raw_scores=score_layout(logical,capacity,padded,shared,row_major)
      probabilities=np.full((16,capacity,padded),np.nan,np.float16)
      self.assertEqual(lib.normalize(capacity,active,total,padded,ptr(raw_scores),ptr(probabilities)),0)
      expected_prob=probability_oracle(logical,capacity,active,total,padded,shared,row_major)
      np.testing.assert_allclose(probabilities,expected_prob,rtol=2e-3,atol=2e-5)

      raw=np.full((16,capacity,128),np.nan,np.float32)
      raw[:,:active]=rng.normal(size=(16,active,128)).astype(np.float32)
      # Convert logical rows to the NPU result's capacity-tiled layout.
      tiled=score_layout(raw,capacity,128,shared,row_major)
      out=np.full((16,capacity,128),np.nan,np.float32)
      self.assertEqual(lib.unpack_result(capacity,active,padded,ptr(tiled),ptr(out)),0)
      if token_major:out=out.reshape(capacity,16,128).transpose(1,0,2)
      expected_out=np.zeros((16,capacity,128),np.float32)
      expected_out[:,:active]=raw[:,:active]
      np.testing.assert_array_equal(out.view(np.uint32),expected_out.view(np.uint32))
      self.assertTrue(np.all(out[:,active:]==0))

  def test_pack_rejects_nonfinite_valid_inputs_and_bad_geometry(self):
    context=128
    lib=compile_helper(context)
    lib.set_pool(0)
    q=np.ones((16,17,128),np.float32)
    cache=np.ones((2,1,8,context,128),np.float16)
    qa=np.empty((16,17,128),np.float16)
    ka=np.empty((16,128,128),np.float16)
    va=np.empty_like(ka)
    q[3,4,7]=np.nan
    self.assertEqual(lib.pack(17,17,62,64,ptr(q),ptr(cache),ptr(qa),ptr(ka),ptr(va)),1)
    q.fill(1)
    cache[0,0,1,5,9]=np.inf
    self.assertEqual(lib.pack(17,17,62,64,ptr(q),ptr(cache),ptr(qa),ptr(ka),ptr(va)),2)
    cache.fill(1)
    cache[1,0,1,5,9]=np.nan
    self.assertEqual(lib.pack(17,17,62,64,ptr(q),ptr(cache),ptr(qa),ptr(ka),ptr(va)),3)
    cache.fill(1)
    for capacity,active,total,padded in ((0,1,1,64),(17,18,18,64),(17,2,1,64),(17,1,65,64),(17,1,1,32),(17,1,1,256)):
      self.assertEqual(lib.pack(capacity,active,total,padded,ptr(q),ptr(cache),ptr(qa),ptr(ka),ptr(va)),10)

  def test_inactive_nan_and_future_cache_poison_do_not_change_outputs(self):
    capacity,active,total,padded,context=65,17,62,64,64
    lib=compile_helper(context)
    lib.set_pool(0)
    q=np.ones((16,capacity,128),np.float32)
    cache=np.ones((2,1,8,context,128),np.float16)
    q[:,active:]=np.nan
    cache[:,:,:,total:]=np.nan
    qa=np.empty((16,capacity,128),np.float16)
    ka=np.empty((16,padded,128),np.float16)
    va=np.empty_like(ka)
    self.assertEqual(lib.pack(capacity,active,total,padded,ptr(q),ptr(cache),ptr(qa),ptr(ka),ptr(va)),0)


class TestRockchipAttentionRuntimeContract(unittest.TestCase):
  def test_submission_errors_and_host_errors_remain_distinct(self):
    import threading
    from tinygrad.runtime.support.rockchip_attention import RockchipAttentionProgram
    for code in (0,1,2,3,4):
      runtime=RockchipAttentionProgram.__new__(RockchipAttentionProgram)
      runtime.heads,runtime.kv_heads,runtime.dim=16,8,128
      runtime.closed=False
      runtime.owned,runtime.plans=[],[]
      runtime.npu=SimpleNamespace(lock=threading.Lock(),check_healthy=lambda:None,submit_count=0,task_count=0,poisoned=False)
      submits,tasks=(2,32) if code==0 else (0,0) if code in (1,2) else (1,16)
      runtime.status=(ctypes.c_uint64*3)(submits,tasks,5)
      runtime.fxn=lambda *_,_code=code:_code
      try:
        if code:
          with self.assertRaises(ValueError if code in (1,2) else RuntimeError):runtime(0,0,0,0,0)
        else:self.assertGreaterEqual(runtime(0,0,0,0,0,wait=True),0)
        self.assertEqual(runtime.npu.poisoned,code==3)
        self.assertEqual((runtime.npu.submit_count,runtime.npu.task_count),(submits,tasks))
      finally:runtime.closed=True

  def test_configuration_requires_explicit_cpu_hook_and_npu_phase(self):
    from tinygrad.llm.kernels.rockchip import configure
    for options in ({'npu_prefill_attention':1},{'npu_prefill_attention':True},
                    {'npu_prefill_attention':True,'cpu_attention':True,'prefill':'off'}):
      with self.assertRaises(ValueError):configure(SimpleNamespace(blk=[]),**options)

  def test_configuration_rejects_tile_width_that_corrupts_mixed_precision(self):
    from tinygrad.llm.kernels.rockchip import configure
    options={'npu_prefill_attention':True,'cpu_attention':True,'prefill':'on','decode':'on','precision':'int8'}
    with self.assertRaisesRegex(ValueError,'column tiles <=24384'):
      configure(SimpleNamespace(blk=[]),tile_columns=24576,**options)
    for width in (4096,24384):
      self.assertEqual(configure(SimpleNamespace(blk=[]),tile_columns=width,**options)['tile_columns'],width)
    self.assertEqual(configure(SimpleNamespace(blk=[]),tile_columns=24576,**{**options,'precision':'fp16'})['tile_columns'],24576)
    self.assertEqual(configure(SimpleNamespace(blk=[]),tile_columns=24576)['tile_columns'],24576)

  def test_static_and_symbolic_metadata_schedule_without_opening_npu(self):
    from tinygrad import Device, Tensor, UOp, dtypes
    from tinygrad.llm.kernels.rockchip_causal_attention import prefill_attention
    from tinygrad.uop.ops import Ops
    query=Tensor.empty(1,16,128,128,device='CPU',dtype=dtypes.float).realize()
    cache=Tensor.empty(2,1,8,512,128,device='CPU',dtype=dtypes.half).realize()
    length=UOp.variable('attention_static_test_length',128,512).bind(128)
    rows=UOp.variable('attention_static_test_rows',1,128).bind(17)
    cases=((query,128),(query,length),(query[:,:,:rows],128))
    for q,total in cases:
      with self.subTest(query_shape=q.shape,length=total):
        linear,_=Tensor.linear_with_vars(prefill_attention(q,cache,total))
        self.assertTrue(any(call.op is Ops.CALL and call.body.op is Ops.PROGRAM
          and isinstance(binary:=call.body.src[3].arg,bytes) and binary[:4]==b'RNPA' for call in linear.src))
    self.assertNotIn('ROCKCHIP',Device._opened_devices)

  def test_bad_descriptor_does_not_open_npu(self):
    import struct
    from tinygrad import Device
    from tinygrad.device import TinyELF
    from tinygrad.llm.kernels.rockchip_cpu import _renderer
    from tinygrad.runtime.support.rockchip_attention import ATTENTION_MAGIC, RockchipAttentionProgram
    cpu=Device['CPU']
    base=[ATTENTION_MAGIC,16,8,128,512,128,0,1]
    for index,value in [(1,12),(4,8192),(5,0),(6,16),(7,2)]:
      fields=base.copy()
      fields[index]=value
      obj=TinyELF(struct.pack('<8I',*fields),'rockchip_prefill_attention',_renderer().target,())
      with patch('tinygrad.runtime.support.rockchip_attention.Device') as devices,self.assertRaises(ValueError):
        cpu.runtime(obj)
      devices.__getitem__.assert_not_called()
    with self.assertRaises(ValueError):RockchipAttentionProgram(cpu,TinyELF(b'bad','attention',_renderer().target,()))

  def test_integrated_source_entry_checks_bounds_without_device_access(self):
    from tinygrad import Device
    from tinygrad.device import TinyELF
    from tinygrad.llm.kernels.rockchip_cpu import _renderer
    from tinygrad.runtime.support.rockchip_attention import RockchipAttentionProgram
    runtime=RockchipAttentionProgram.__new__(RockchipAttentionProgram)
    runtime.heads,runtime.kv_heads,runtime.dim=16,8,128
    runtime.context,runtime.capacity,runtime.pool,runtime.plans,runtime.closed,runtime.owned=512,128,None,[],True,[]
    runtime.fast_softmax=True
    runtime.row_major,runtime.token_major=True,True
    runtime.status=(ctypes.c_uint64*3)()
    runtime.npu=SimpleNamespace(fd=SimpleNamespace(fd=-1),domain=0)
    for i,name in enumerate(['qa','ka','va','scores','probabilities','result']):
      setattr(runtime,name,SimpleNamespace(addr=4096*(i+1),meta=SimpleNamespace(obj_addr=i+1)))
    renderer=_renderer()
    program=Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(runtime.make_source()),'rockchip_prefill_attention',renderer.target,()))
    program.fxn.restype=ctypes.c_int
    for active,total in [(0,128),(1,513),(129,512),(17,16)]:
      rows,length=ctypes.c_int(active),ctypes.c_int(total)
      self.assertEqual(program.fxn(None,None,None,ctypes.byref(length),ctypes.byref(rows)),1)
      self.assertEqual(list(runtime.status),[0,0,0])


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
class TestRockchipCausalAttentionHardware(unittest.TestCase):
  def test_dynamic_prefill_native_replay_and_invalid_input_recovery(self):
    from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
    from tinygrad.llm.kernels.rockchip_causal_attention import prefill_attention
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    from tinygrad.llm.kernels.rockchip_pool import attention_pool
    pool=attention_pool()
    rng=np.random.default_rng(20261029)
    q=Tensor.empty(1,16,128,128,device='CPU',dtype=dtypes.float32)
    kv=Tensor.empty(2,1,8,512,128,device='CPU',dtype=dtypes.float16)
    run=TinyJit(lambda query,cache,active,total:
      prefill_attention(query[:,:,:active],cache,total,fast_softmax=True,pool=pool).pad_to((1,16,128,128)).realize())
    active_var,total_var=UOp.variable('attention_rows',1,128),UOp.variable('attention_length',1,512)
    def execute(active,total):return run(q,kv,active_var.bind(active),total_var.bind(total))
    requests=((17,62),(128,128),(1,64),(17,190),(128,512),(1,512),(65,256))
    for index,(active,total) in enumerate(requests):
      qv=rng.normal(0,.25,(1,16,128,128)).astype(np.float32)
      qv[:,:,active:]=np.nan
      kvv=rng.normal(0,.2,(2,1,8,512,128)).astype(np.float16)
      kvv[:,:,:,total:]=np.nan
      q.assign(Tensor(qv,device='CPU')).realize()
      kv.assign(Tensor(kvv,device='CPU')).realize()
      device=Device['ROCKCHIP']
      before=device.submit_count
      actual=execute(active,total).numpy()
      after=device.submit_count
      self.assertEqual(after-before,2)
      q16=qv[:,:,:active].astype(np.float16).astype(np.float32)
      key=np.repeat(kvv[0,0,:,:total].astype(np.float32),2,axis=0)
      value=np.repeat(kvv[1,0,:,:total].astype(np.float32),2,axis=0)
      scores=q16@key.transpose(0,2,1)/np.float32(np.sqrt(128))
      for row in range(active):scores[...,row,total-active+row+1:]=-np.inf
      probs=np.exp(scores-scores.max(axis=-1,keepdims=True))
      probs/=probs.sum(axis=-1,keepdims=True)
      expected=np.zeros((1,16,128,128),np.float32)
      expected[0,:,:active]=(probs.astype(np.float16).astype(np.float32)@value)[0]
      np.testing.assert_allclose(actual,expected,rtol=1e-2,atol=3e-4)
      if index==1:run.captured=RockchipCapturedJit(run.captured)
    self.assertGreater(run.captured.native_runs,0)
    self.assertIsNone(run.captured.reason)

    device=Device['ROCKCHIP']
    submits=device.submit_count
    qv=np.ones((1,16,128,128),np.float32)
    kvv=np.ones((2,1,8,512,128),np.float16)
    qv[0,0,0,0]=np.nan
    q.assign(Tensor(qv,device='CPU')).realize()
    kv.assign(Tensor(kvv,device='CPU')).realize()
    with self.assertRaises(ValueError):execute(1,64)
    self.assertEqual(device.submit_count,submits)
    self.assertFalse(device.poisoned)
    qv.fill(1)
    for region in ((0,0),(1,0)):
      kvv.fill(1)
      kvv[region[0],0,0,0,0]=np.nan
      q.assign(Tensor(qv,device='CPU')).realize()
      kv.assign(Tensor(kvv,device='CPU')).realize()
      with self.assertRaises(ValueError):execute(1,64)
      self.assertEqual(device.submit_count,submits)
    kvv.fill(1)
    q.assign(Tensor(qv,device='CPU')).realize()
    kv.assign(Tensor(kvv,device='CPU')).realize()
    self.assertTrue(np.isfinite(execute(1,64).numpy()).all())
    self.assertGreater(run.captured.native_runs,0)


if __name__=='__main__':unittest.main()
