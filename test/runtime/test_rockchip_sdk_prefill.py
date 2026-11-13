"""SDK prefill requires matching command, input and output layouts."""
import os, struct, unittest
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import ceildiv, round_up
from tinygrad.llm.kernels.rockchip_int8 import quantize, dequantize_tiles, raw_int8_matmul
from tinygrad.runtime.support.rockchip_int8 import SDK_PREFILL_SHAPES, int8_sdk_body, int8_sdk_bodies, sdk_prefill_supported


def unpack(raw,rows,columns):
  n=round_up(columns,32)
  width=round_up(ceildiv(n,3),32)
  parts=[(start,min(width,n-start)) for start in range(0,n,width)]
  if len(parts)!=3:parts=[(0,n)]
  return np.concatenate([raw[start*rows:(start+count)*rows].reshape(count//4,rows,4).transpose(1,0,2).reshape(rows,count)
                         for start,count in parts],axis=1)[:,:columns]


class TestSDKPrefill(unittest.TestCase):
  def test_lfm_k512_commands_match_every_ordered_capture_word(self):
    import json
    from pathlib import Path
    from tinygrad.runtime.support.rockchip_int8 import int8_sdk_bodies,sdk_prefill_supported
    fixture=json.loads((Path(__file__).parent/'fixtures/lfm_k512_sdk_prefill.json').read_text())
    addresses=fixture['addresses']
    self.assertTrue(sdk_prefill_supported(128,1024,512))
    for columns,expected in fixture['cases'].items():
      width=int(columns)
      self.assertEqual(int8_sdk_bodies(128,width,512,addresses['lhs'],addresses['rhs'],addresses['dst'],width),
                       (tuple(expected),))
    for width,stride in ((384,384),(352,320)):
      with self.assertRaises(ValueError):int8_sdk_bodies(128,width,512,0,0,0,stride)

  def test_sdk_quantizer_overwrites_every_byte_before_opt_in_copy_alias(self):
    from tinygrad import Device,UOp,dtypes
    from tinygrad.device import TinyELF
    from tinygrad.helpers import Context
    from tinygrad.uop.ops import Ops
    from tinygrad.runtime.ops_cpu import CPUProgram
    from tinygrad.llm.kernels.rockchip_cpu import _renderer
    from tinygrad.llm.kernels.rockchip_int8 import _quantize
    from tinygrad.llm.kernels.rockchip_replay import _full_write
    for sdk in (False,True):
      for rows,depth,padded in ((1,1024,1024),(128,1024,1024),(128,2048,2048),(17,65,512)):
        with self.subTest(sdk=sdk,rows=rows,depth=depth,padded=padded):
          values=np.random.default_rng(rows+depth).normal(size=(rows,depth)).astype(np.float32)
          values[0]=0
          if rows>1:values[1,0]=np.inf
          x=Tensor(values,device='CPU').realize()
          expected=quantize(x,padded,sdk_prefill=sdk)
          expected_bytes=expected[0].numpy().copy()
          expected_scales=expected[1].numpy().view(np.uint32).copy()
          size=len(expected_bytes)
          params=[UOp.param(i,dt,shape,device='CPU') for i,(dt,shape) in enumerate(
            ((dtypes.uint8,(size,)),(dtypes.float,(rows,depth)),(dtypes.float,(rows,))))]
          program=_quantize(*params,rows,depth,padded,sdk)
          binary=next(x.arg for x in program.src if x.op is Ops.BINARY)
          runtime=CPUProgram(Device['CPU'],TinyELF(binary,program.src[0].arg.function_name,_renderer().target,()))
          for poison in (0xa5,0x5a):
            packet=np.full(size,poison,np.uint8)
            scales=np.full(rows,poison*0x01010101,np.uint32)
            runtime(packet.ctypes.data,values.ctypes.data,scales.ctypes.data)
            np.testing.assert_array_equal(packet,expected_bytes)
            np.testing.assert_array_equal(scales,expected_scales)
          call=program.call(UOp.new_buffer('CPU',size,dtypes.uint8),UOp.new_buffer('CPU',rows*depth,dtypes.float),
                            UOp.new_buffer('CPU',rows,dtypes.float))
          with Context(ONPU_REPLAY_EXTRA_PACK_ALIAS=0):self.assertEqual(_full_write(call,0,size),not sdk)
          with Context(ONPU_REPLAY_EXTRA_PACK_ALIAS=1):
            self.assertTrue(_full_write(call,0,size))
            self.assertFalse(_full_write(call,0,size+64))
            self.assertFalse(_full_write(call,1,values.nbytes))

  def test_spark_half_row_commands_match_every_ordered_sdk_capture_word(self):
    import json
    from pathlib import Path
    fixture=json.loads((Path(__file__).parent/'fixtures/spark_m64_k2048_sdk_prefill.json').read_text())
    addresses={(0x201,0x1070):0x100000,(0x201,0x1110):0x200000,(0x1001,0x4020):0x800000}
    for columns,words in fixture['cases'].items():
      width=int(columns)
      with self.subTest(columns=width):
        expected=tuple((target<<48)|(addresses.get((target,address),value)<<16)|address for target,address,value in words)
        bodies=int8_sdk_bodies(64,width,2048,0x100000,0x200000,0x800000,width)
        self.assertEqual(len(bodies),1)
        self.assertEqual(len(bodies[0]),108)
        self.assertEqual(bodies[0],expected)
        with self.assertRaises(ValueError):int8_sdk_bodies(64,width,2048,0,0,0,width+32)

  def test_quantization_writes_blocked_codes_without_changing_scales_or_tails(self):
    rng=np.random.default_rng(64016)
    for rows,depth,padded in ((64,1024,1024),(128,1023,1024),(64,1031,2048),(17,65,512),(5,17,512)):
      values=rng.normal(0,.3,(rows,depth)).astype(np.float32)
      values[0]=0
      x=Tensor(values,device='CPU').realize()
      ordinary,sa=quantize(x,padded)
      blocked,sb=quantize(x,padded,sdk_prefill=True)
      a,b=ordinary.numpy(),blocked.numpy()
      header=round_up(rows*4,64)
      np.testing.assert_array_equal(a[:header],b[:header])
      np.testing.assert_array_equal(sa.numpy().view(np.uint32),sb.numpy().view(np.uint32))
      restored=b[header:].reshape(padded//16,rows,16).transpose(1,0,2).reshape(rows,padded)
      np.testing.assert_array_equal(restored,a[header:].reshape(rows,padded))

  def test_blocked_quantization_keeps_nonfinite_rows_invalid_and_zeroed(self):
    for bad in (float('nan'),float('inf'),float('-inf')):
      values=np.ones((17,65),dtype=np.float32)
      values[1,0]=bad
      values[16,64]=bad
      ordinary,sa=quantize(Tensor(values,device='CPU'),512)
      blocked,sb=quantize(Tensor(values,device='CPU'),512,sdk_prefill=True)
      a,b=ordinary.numpy(),blocked.numpy()
      header=round_up(17*4,64)
      np.testing.assert_array_equal(a[:header],b[:header])
      np.testing.assert_array_equal(sa.numpy().view(np.uint32),sb.numpy().view(np.uint32))
      restored=b[header:].reshape(32,17,16).transpose(1,0,2).reshape(17,512)
      np.testing.assert_array_equal(restored,a[header:].reshape(17,512))

  def test_mixed_output_tiles_restore_exact_float_bits_with_partial_columns(self):
    rng=np.random.default_rng(64004)
    for columns,counts,flags in ((2111,[64],(True,)),(3057,[64,17],(True,False)),(3072,[128],(True,)),
                                (3071,[17,9],(True,True)),(3057,[64,17],(False,True))):
      n=round_up(columns,32)
      width=round_up(ceildiv(n,3),32)
      parts=[(start,min(width,n-start)) for start in range(0,n,width)]
      raw=[]
      scales=[]
      expected=[]
      sw=rng.uniform(.001,.1,columns).astype(np.float32)
      for rows,blocked in zip(counts,flags):
        values=rng.integers(-1000000,1000000,(rows,n),dtype=np.int32)
        sa=rng.uniform(.001,.1,rows).astype(np.float32)
        data=np.concatenate([(values[:,start:start+count].reshape(rows,count//4,4).transpose(1,0,2).ravel()
                              if blocked else values[:,start:start+count].ravel()) for start,count in parts])
        raw.append(Tensor(data,device='CPU'))
        scales.append(Tensor(sa,device='CPU'))
        expected.append((values[:,:columns].astype(np.float32)*sa[:,None])*sw[None,:])
      actual=dequantize_tiles(raw,scales,Tensor(sw,device='CPU'),counts,columns,sdk_tiles=flags).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),np.concatenate(expected).view(np.uint32))

  def test_only_qualified_commands_and_descriptor_are_accepted(self):
    from tinygrad.runtime.ops_rockchip import RockchipMatmul
    for m,n,k in SDK_PREFILL_SHAPES:
      self.assertTrue(sdk_prefill_supported(m,n,k))
      plan=RockchipMatmul(SimpleNamespace(programs=set()),SimpleNamespace(lib=struct.pack('<9I',m,n,k,3,0,round_up(m*4,64),0,1,1)))
      self.assertEqual(plan.sdk_prefill,1)
      for _,width in plan.parts:
        bodies=int8_sdk_bodies(m,width,k,0,0,0,width)
        split=m==128 and k in (1536,2048)
        self.assertEqual(len(bodies),2 if split else 1)
        for index,words in enumerate(bodies):
          rows=(44 if index==0 else 84) if k==1536 else 64 if split else m
          values={(w>>48,w&0xffff):(w>>16)&0xffffffff for w in words}
          self.assertEqual(values[0x201,0x1010],32)
          self.assertEqual(values[0x201,0x1020],(rows<<16)|1)
          self.assertEqual(values[0x1001,0x4024],m*16)
          if k==896 or (m,width,k) in ((128,2048,1024),(128,2752,1024),(128,2688,1024),(128,3072,1024)) or split:
            self.assertEqual(values[0x201,0x107c],m if split else m*4)
            self.assertEqual(values[0x801,0x3014],rows-1)
            self.assertEqual(values[0x1001,0x405c],rows-1)
    for shape in ((64,6144,4096),(128,4096,2048),(64,2048,1024),(1,3072,1024)):
      self.assertFalse(sdk_prefill_supported(*shape))
      m,n,k=shape
      with self.assertRaises(ValueError):
        RockchipMatmul(SimpleNamespace(programs=set()),SimpleNamespace(lib=struct.pack('<9I',m,n,k,3,0,round_up(m*4,64),0,1,1)))
    with self.assertRaises(ValueError):int8_sdk_body(64,640,1024,0,0,0,640)

  def test_minicpm4_native_commands_match_every_ordered_capture_word(self):
    import json
    from pathlib import Path
    fixture=json.loads((Path(__file__).parent/'fixtures/minicpm4_sdk_prefill.json').read_text())
    for case in fixture['cases']:
      m,n,k=case['shape']
      registers=case['registers']
      values={(r['target'],r['address']):r['value'] for r in registers}
      words=int8_sdk_body(m,n,k,values[0x201,0x1070],values[0x201,0x1110],values[0x1001,0x4020],n)
      actual=[{'target':w>>48,'address':w&0xffff,'value':(w>>16)&0xffffffff} for w in words]
      self.assertEqual(actual,registers)

  def test_shared_payload_offset_is_bounded_aligned_and_locates_its_scale_header(self):
    from tinygrad import Device,dtypes
    from tinygrad.uop.ops import Ops
    from tinygrad.runtime.ops_rockchip import RockchipMatmul
    rows,columns,depth=17,192,512
    size=round_up(rows*4,64)+rows*depth
    payload=Tensor.empty(size+64,device='CPU',dtype=dtypes.uint8)
    weight=Tensor.empty(columns//32,depth//32,32,32,device='ROCKCHIP',dtype=dtypes.int8)
    for offset in (-64,1,True,2**32,128):
      with self.assertRaises(ValueError):raw_int8_matmul(payload,weight,rows,columns,payload_offset=offset)
    linear,_=raw_int8_matmul(payload,weight,rows,columns,payload_offset=64).linear_with_vars()
    programs=[c.body for c in linear.src if c.op is Ops.CALL and c.body.op is Ops.PROGRAM and c.device=='ROCKCHIP']
    self.assertEqual(len(programs),1)
    plan=RockchipMatmul(SimpleNamespace(programs=set()),SimpleNamespace(lib=programs[0].src[3].arg))
    self.assertEqual(plan.scale_offset,64)
    self.assertEqual(plan.lhs_offset,64+round_up(rows*4,64))
    with self.assertRaises(ValueError):
      RockchipMatmul(SimpleNamespace(programs=set()),SimpleNamespace(lib=struct.pack('<10I',17,192,512,3,0,128,0,1,0,1)))
    self.assertNotIn('ROCKCHIP',Device._opened_devices)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit exclusive NPU hardware test')
  def test_changed_inputs_match_independent_int32_through_native_replay(self):
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng=np.random.default_rng(64027)
    for rows,columns,depth in sorted(SDK_PREFILL_SHAPES):
      weights=rng.integers(-128,128,(columns,depth),dtype=np.int8)
      packed=weights.reshape(columns//32,32,depth//32,32).transpose(0,2,1,3).copy()
      weight=Tensor(packed,device='ROCKCHIP').realize()
      run=TinyJit(lambda payload:raw_int8_matmul(payload,weight,rows,columns,sdk_prefill=True).realize())
      for step in range(7):
        x=rng.integers(-128,128,(rows,depth),dtype=np.int8)
        if step==6:x[:,::2],x[:,1::2]=-128,127
        header=round_up(rows*4,64)
        payload=np.zeros(header+rows*depth,dtype=np.uint8)
        payload[:rows*4]=np.ones(rows,dtype=np.float32).view(np.uint8)
        payload[header:]=x.reshape(rows,depth//16,16).transpose(1,0,2).copy().ravel().view(np.uint8)
        actual=unpack(run(Tensor(payload,device='CPU').realize()).numpy(),rows,columns)
        np.testing.assert_array_equal(actual,x.astype(np.int32)@weights.astype(np.int32).T)
        if step==2:run.captured=RockchipCapturedJit(run.captured)
      self.assertIsNone(run.captured.reason)
      self.assertGreaterEqual(run.captured.native_runs,2)
