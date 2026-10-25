import os, unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.engine.jit import CapturedJit
from tinygrad.helpers import Target
from tinygrad.llm.kernels import rockchip_replay
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit, NativeReplay, UnsupportedReplay, _full_write
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.runtime.ops_rockchip import RockchipBuffer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo


class TestRockchipReplayAliasAnalysis(unittest.TestCase):
  """Exercise the actual alias pass without allocating or opening an NPU device."""
  @staticmethod
  def node(device='CPU',size=4,dtype=dtypes.half,base=None,offset=0):
    node = SimpleNamespace(dtype=dtype,toposort=lambda:[SimpleNamespace(op=Ops.BUFFER,tag='scratch')])
    buf = SimpleNamespace(base=object() if base is None else base,offset=offset,nbytes=size*dtype.itemsize,device=device)
    raw = RockchipBuffer(64,64,buf.nbytes,SimpleNamespace(flags=rk.RKNPU_MEM_CACHEABLE)) if device=='ROCKCHIP' else 0
    return node,(buf,raw,f'{device}_{id(node)}')

  @staticmethod
  def program(size=4,*,count=None,stride=1,start=0,gate=False,name='replay_test_writer',opaque=False,index_fxn=None):
    out,inp = UOp.param(0,dtypes.half,size),UOp.param(1,dtypes.half,size)
    row = UOp.range(size if count is None else count,0,dtype=dtypes.int)
    index = row*stride+start if index_fxn is None else index_fxn(row)
    store = out[index].store(inp[row].load(),gate=row<2 if gate else None)
    sink = UOp.sink(out,inp,arg=KernelInfo(name)) if opaque else store.end(row).sink(arg=KernelInfo(name))
    info = replace(ProgramInfo.from_sink(sink,Target('CPU')),outs=(0,),ins=(1,))
    return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,inp))))

  @staticmethod
  def call(body,*nodes):
    return SimpleNamespace(op=Ops.CALL,body=body,src=(body,*nodes),device='CPU',arg=SimpleNamespace(aux=None))

  def analyze(self,calls,*bindings,borrow=False):
    replay = NativeReplay.__new__(NativeReplay)
    replay.calls,replay.roots,replay.borrow_outputs = calls,[],borrow
    buffers = {id(node):binding for node,binding in bindings}
    with patch.object(NativeReplay,'buffer',lambda _,node:buffers[id(node)]): return replay.copy_aliases()

  def test_partial_producer_keeps_upload_and_old_bytes(self):
    cpu,npu,source = self.node(),self.node('ROCKCHIP'),self.node()
    calls = [self.call(self.program(),cpu[0],source[0]),self.call(self.program(count=1),cpu[0],source[0]),
             self.call(UOp(Ops.STORE),npu[0],cpu[0])]
    aliases,uploads,downloads = self.analyze(calls,cpu,npu,source)
    self.assertEqual((aliases,uploads,downloads),({},set(),set()))
    host,device = np.array([1,2,3,4],dtype=np.float16),np.zeros(4,dtype=np.float16)
    (device if aliases else host)[0] = 9
    if 2 not in uploads: device[:] = host
    np.testing.assert_array_equal(device,[9,2,3,4])

  def test_full_producer_still_elides_upload(self):
    cpu,npu,source = self.node(),self.node('ROCKCHIP'),self.node()
    for name,opaque in (('replay_test_writer',False),('rockchip_swiglu_pack',True),('rockchip_quantize',True),
                        ('rockchip_int8_ffn_pack',True)):
      with self.subTest(name=name):
        calls = [self.call(self.program(name=name,opaque=opaque),cpu[0],source[0]),self.call(UOp(Ops.STORE),npu[0],cpu[0])]
        aliases,uploads,downloads = self.analyze(calls,cpu,npu,source)
        key = (cpu[1][0].base,0,8)
        self.assertEqual((aliases,uploads,downloads),({(0,key):npu[1][2]},{1},set()))

  def test_partial_future_reuse_keeps_original_producer(self):
    cpu,npu,source,result = self.node(),self.node('ROCKCHIP'),self.node(),self.node()
    for count in (1,4):
      with self.subTest(count=count):
        calls = [self.call(self.program(),cpu[0],source[0]),self.call(UOp(Ops.STORE),npu[0],cpu[0]),
                 self.call(self.program(count=count),cpu[0],source[0]),self.call(self.program(),result[0],cpu[0])]
        aliases,uploads,_ = self.analyze(calls,cpu,npu,source,result)
        self.assertEqual(uploads,{1} if count==4 else set())
        self.assertEqual(len(aliases),int(count==4))

  def test_masks_holes_and_wrong_extent_are_not_full_writes(self):
    for options,nbytes in (({'count':1},8),({'count':2,'stride':2},8),({'gate':True},8),
                           ({'count':3,'start':1},8),({},16),({'opaque':True},8),
                           ({'name':'rockchip_swiglu_pack','opaque':True},16)):
      with self.subTest(options=options,nbytes=nbytes):
        self.assertFalse(_full_write(self.call(self.program(**options)),0,nbytes))

  def test_wrapping_index_arithmetic_is_not_a_full_write(self):
    program = self.program(index_fxn=lambda row:(row.cast(dtypes.int8)+UOp.const(127,dtypes.int8)).cast(dtypes.int)-127)
    self.assertFalse(_full_write(self.call(program),0,8))

  def test_generated_cpu_pack_remains_eligible(self):
    run = TinyJit(lambda x:x.half().pad(((0,0),(0,3))).reshape(2,4,2).permute(1,0,2).contiguous().realize())
    for _ in range(3): run(Tensor(np.arange(10,dtype=np.float32).reshape(2,5),device='CPU').realize())
    calls = [call.without_after for call in run.captured.linear.src]
    self.assertEqual(len(calls),1)
    self.assertTrue(_full_write(calls[0],0,32))
    self.assertFalse(_full_write(calls[0],0,64))

  def test_internal_cpu_packers_remain_eligible(self):
    from tinygrad.llm.kernels.rockchip_cpu import swiglu_pack
    from tinygrad.llm.kernels.rockchip_int8 import quantize, ffn_pack
    outputs = [swiglu_pack(Tensor.ones(2,64,device='CPU'),32,2),
               quantize(Tensor.ones(2,32,device='CPU'),512)[0],
               ffn_pack(Tensor.ones(2048,device='CPU',dtype=dtypes.int32),Tensor.ones(2,device='CPU'),
                        Tensor.ones(1024,device='CPU'),2,512)[0]]
    for output,name in zip(outputs,('rockchip_swiglu_pack','rockchip_quantize','rockchip_int8_ffn_pack')):
      with self.subTest(name=name):
        linear,_ = output.linear_with_vars()
        calls = [call.without_after for call in linear.src if call.without_after.body.op is Ops.PROGRAM
                 and call.without_after.body.src[0].arg.function_name==name]
        self.assertEqual(len(calls),1)
        self.assertTrue(_full_write(calls[0],0,output.numel()*output.dtype.itemsize))
        self.assertFalse(_full_write(calls[0],0,output.numel()*output.dtype.itemsize+64))

  def test_same_base_overlapping_future_view_keeps_copy(self):
    cpu,npu,source,result = self.node(),self.node('ROCKCHIP'),self.node(),self.node()
    view = self.node(size=2,base=cpu[1][0].base,offset=2)
    calls = [self.call(self.program(),cpu[0],source[0]),self.call(UOp(Ops.STORE),npu[0],cpu[0]),
             self.call(self.program(size=2),view[0],source[0]),self.call(self.program(),result[0],cpu[0])]
    self.assertEqual(self.analyze(calls,cpu,npu,source,result,view),({},set(),set()))

  def test_download_borrowing_keeps_partial_future_reuse_copy(self):
    cpu,npu,source,result = self.node(),self.node('ROCKCHIP'),self.node(),self.node()
    for count in (None,1):
      with self.subTest(future_partial_write=count):
        calls = [self.call(UOp(Ops.STORE),cpu[0],npu[0]),self.call(self.program(),result[0],cpu[0])]
        if count is not None:
          calls.extend([self.call(self.program(count=count),cpu[0],source[0]),self.call(self.program(),result[0],cpu[0])])
        aliases,uploads,downloads = self.analyze(calls,cpu,npu,source,result,borrow=True)
        self.assertEqual(uploads,set())
        self.assertEqual(downloads,{0} if count is None else set())
        self.assertEqual(len(aliases),int(count is None))


class TestRockchipReplayFallback(unittest.TestCase):
  def test_prefill_variants_reuse_multiple_input_sizes(self):
    model = SimpleNamespace(blk=[],greedy_prefill_jit=TinyJit(lambda x:(x*2+1).realize()))
    configure(model,native_replay=True)
    for rows in (16,32,64,16,32,64,16,32,64,16,64):
      values = np.arange(rows*8,dtype=np.float32).reshape(1,rows,8)
      actual = model.greedy_prefill_jit(Tensor(values,device='CPU').realize()).numpy()
      np.testing.assert_array_equal(actual,values*2+1)

  def test_cpu_only_graph_uses_original_executor(self):
    run = TinyJit(lambda x:(x*2+1).realize())
    for i in range(3): run(Tensor([float(i)],device='CPU').realize())
    run.captured = RockchipCapturedJit(run.captured)
    for i in range(3,7):
      np.testing.assert_array_equal(run(Tensor([float(i)],device='CPU').realize()).numpy(),[i*2+1])
    self.assertEqual(run.captured.reason,'no NPU work')
    self.assertEqual(run.captured.native_runs,0)


class TestRockchipReplayPreparation(unittest.TestCase):
  @staticmethod
  def captured():
    marker = object()
    linear = UOp(Ops.LINEAR, src=())
    wrapped = RockchipCapturedJit(CapturedJit(marker, linear, [], []))
    # Bypass unrelated capture traversal: the tests exercise only call ordering.
    wrapped.__dict__['linear'] = linear
    wrapped.__dict__['_jit_bufs'] = set()
    wrapped.__dict__['_symbolic_ret'] = []
    return wrapped, marker

  def test_warmup_prepares_replay_after_one_ordinary_commit_and_defers_native_execution(self):
    events = []
    commits = []
    wrapped, marker = self.captured()

    class FakeReplay:
      def valid(self, inputs):
        events.append('valid')
        return True

      def __call__(self, inputs, values):
        events.append('native')

    replay = FakeReplay()

    def ordinary(_self, input_uops, var_vals):
      events.append('ordinary')
      commits.append('state')
      return marker

    def prepare(linear, inputs, values, cleaner):
      events.append('prepare')
      return replay

    with patch.object(CapturedJit, '__call__', ordinary), patch.object(rockchip_replay, 'NativeReplay', prepare):
      first = wrapped([], {})
      self.assertIs(first, marker)
      self.assertTrue(wrapped.warmed)
      self.assertIs(wrapped.native, replay)
      self.assertEqual(wrapped.native_runs, 0)
      self.assertEqual(commits, ['state'])
      self.assertEqual(events, ['ordinary', 'prepare'])

      second = wrapped([], {})
      self.assertIs(second, marker)
      self.assertEqual(wrapped.native_runs, 1)
      self.assertEqual(commits, ['state'])
      self.assertEqual(events, ['ordinary', 'prepare', 'valid', 'native'])

  def test_warmup_constructor_failures_return_first_result_then_fallback_once_per_later_call(self):
    failures = (UnsupportedReplay('unsupported replay'), KeyError('missing binding'), AssertionError('bad invariant'))
    for failure in failures:
      with self.subTest(failure=type(failure).__name__):
        events, commits = [], []
        wrapped, marker = self.captured()

        def ordinary(_self, input_uops, var_vals):
          events.append('ordinary')
          commits.append('state')
          return marker

        def prepare(linear, inputs, values, cleaner):
          events.append('prepare')
          raise failure

        with patch.object(CapturedJit, '__call__', ordinary), patch.object(rockchip_replay, 'NativeReplay', prepare):
          first = wrapped([], {})
          self.assertIs(first, marker)
          self.assertEqual(commits, ['state'])
          self.assertEqual(events, ['ordinary', 'prepare'])
          self.assertEqual(wrapped.native_runs, 0)
          self.assertEqual(wrapped.reason, str(failure))

          second = wrapped([], {})
          self.assertIs(second, marker)
          self.assertEqual(commits, ['state', 'state'])
          self.assertEqual(events, ['ordinary', 'prepare', 'ordinary'])
          self.assertEqual(wrapped.native_runs, 0)

  def test_invalid_native_rebuild_failure_falls_back_once_without_replaying_warmup(self):
    events, commits = [], []
    wrapped, marker = self.captured()

    class InvalidReplay:
      def valid(self, inputs):
        events.append('invalid')
        return False

      def __call__(self, inputs, values):
        events.append('native')

    replay = InvalidReplay()
    attempts = []

    def ordinary(_self, input_uops, var_vals):
      events.append('ordinary')
      commits.append('state')
      return marker

    def prepare(linear, inputs, values, cleaner):
      attempts.append('prepare')
      events.append('prepare')
      if len(attempts) == 1:
        return replay
      raise UnsupportedReplay('invalidated replay cannot rebuild')

    with patch.object(CapturedJit, '__call__', ordinary), patch.object(rockchip_replay, 'NativeReplay', prepare):
      self.assertIs(wrapped([], {}), marker)
      self.assertEqual(commits, ['state'])
      self.assertEqual(events, ['ordinary', 'prepare'])
      self.assertEqual(wrapped.native_runs, 0)

      result = wrapped([], {})
      self.assertIs(result, marker)
      self.assertEqual(commits, ['state', 'state'])
      self.assertEqual(events, ['ordinary', 'prepare', 'invalid', 'prepare', 'ordinary'])
      self.assertEqual(wrapped.native_runs, 0)
      self.assertEqual(wrapped.reason, 'invalidated replay cannot rebuild')


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
@patch.dict(os.environ,ONPU_REPLAY_BORROW_OUTPUTS='1')
class TestRockchipReplayBorrow(unittest.TestCase):
  def test_private_and_returned_outputs_rebind_and_survive_scratch_release(self):
    from tinygrad import UOp
    from tinygrad.helpers import getenv
    from tinygrad.llm.kernels.rockchip import pack_weight, packed_matmul
    rng = np.random.default_rng(205)
    weight_data = rng.normal(0,.1,(96,64)).astype(np.float16)
    weight = pack_weight(Tensor(weight_data,device='CPU')).realize()
    inputs = [rng.normal(0,.1,(8,64)).astype(np.float32) for _ in range(10)]
    for held in (False,True):
      reference = []
      for enabled in ('0','1'):
        def calculation(x,position):
          raw = packed_matmul(x,weight,96,raw_output=True).realize()
          out = (raw.reshape(24,8,4).permute(1,0,2).reshape(8,96)+Tensor(position)*.125).realize()
          return (out,raw.realize()) if held else out
        run = TinyJit(calculation)
        with patch.dict(os.environ,ONPU_REPLAY_BORROW_OUTPUTS=enabled):
          getenv.cache_clear()
          for position,data in enumerate(inputs):
            result = run(Tensor(data,device='CPU').realize(),UOp.variable('borrow_position',0,16).bind(position))
            out = result[0].numpy() if held else result.numpy()
            expected = data.astype(np.float16).astype(np.float32)@weight_data.astype(np.float32).T+np.float32(position*.125)
            np.testing.assert_allclose(out,expected,atol=2e-5,rtol=3e-4)
            if enabled=='0': reference.append(out.copy())
            else: np.testing.assert_array_equal(out.view(np.uint32),reference[position].view(np.uint32))
            if held:
              raw_expected = (expected-np.float32(position*.125)).reshape(8,24,4).transpose(1,0,2).ravel()
              np.testing.assert_allclose(result[1].numpy(),raw_expected,atol=2e-5,rtol=3e-4)
            if position==1:run.captured = RockchipCapturedJit(run.captured)
            if position==6:
              self.assertGreater(run.captured.native_runs,0)
              native = run.captured.native
              self.assertEqual(native.driver_from_sites,1)
              self.assertEqual(len(native.output_copies),int(enabled=='1' and not held))
              run.captured.free_intermediates()
          self.assertIsNone(run.captured.reason)
          self.assertGreater(run.captured.native_runs,0)
        getenv.cache_clear()

  def test_multiple_cpu_readers_keep_the_copy(self):
    from tinygrad.llm.kernels.rockchip import pack_weight, packed_matmul
    rng = np.random.default_rng(206)
    weights = rng.normal(0,.1,(96,64)).astype(np.float16)
    weight = pack_weight(Tensor(weights,device='CPU')).realize()
    for partial_write in (False,True):
      def calculation(x):
        raw = packed_matmul(x,weight,96,raw_output=True).realize()
        first = (raw+1).realize()
        if partial_write:raw[:32].assign(Tensor.zeros(32,device='CPU')).realize()
        second = (raw*2).realize()
        return first,second
      run = TinyJit(calculation)
      for index in range(6):
        data = rng.normal(0,.1,(1,64)).astype(np.float32)
        expected = (data.astype(np.float16).astype(np.float32)@weights.astype(np.float32).T).ravel()
        first,second = run(Tensor(data,device='CPU').realize())
        np.testing.assert_allclose(first.numpy(),expected+1,atol=2e-5,rtol=3e-4)
        if partial_write:expected[:32]=0
        np.testing.assert_allclose(second.numpy(),expected*2,atol=2e-5,rtol=3e-4)
        if index==1:run.captured = RockchipCapturedJit(run.captured)
      self.assertIsNone(run.captured.reason)
      self.assertGreater(run.captured.native_runs,0)
      self.assertEqual(run.captured.native.submits,1)
      self.assertEqual(len(run.captured.native.output_copies),0)


if __name__=='__main__': unittest.main()
