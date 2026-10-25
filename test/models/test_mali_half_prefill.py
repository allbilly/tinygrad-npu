"""Exact FP16 projection bits, padding and runtime launches across JIT replay."""
import os, unittest
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, function, nn
from tinygrad.dtype import AddrSpace
from tinygrad.llm.kernels import mali_half_prefill
from tinygrad.llm.kernels.mali import MaliLinear


@contextmanager
def launches():
  from tinygrad.runtime.ops_cl import CLProgram
  initialize,invoke=CLProgram.__init__,CLProgram.__call__
  records=[]
  def init(program,device,obj):
    initialize(program,device,obj)
    program._half_prefill_test_name=obj.name
  def call(program,*args,**kwargs):
    if getattr(program,'_half_prefill_test_name','').startswith('mali_half_prefill_'):
      scalars={name:value for value,(name,space,_,_) in zip(args,program.signature) if space is AddrSpace.ALU}
      records.append((kwargs['global_size'],kwargs['local_size'],scalars))
    return invoke(program,*args,**kwargs)
  with patch.object(CLProgram,'__init__',init),patch.object(CLProgram,'__call__',call):yield records


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestMaliHalfPrefill(unittest.TestCase):
  def test_compiler_fingerprint_rejects_an_unknown_source(self):
    self.assertEqual(mali_half_prefill.prepare('CL')['prepared_kernels'],2)
    mali_half_prefill._sources.pop(('CL',2048))
    try:
      with patch.dict(mali_half_prefill._QUALIFIED,{2048:'0'*64}):
        self.assertEqual(mali_half_prefill.prepare('CL')['prepared_kernels'],1)
        self.assertNotIn(('CL',2048),mali_half_prefill._sources)
    finally:
      self.assertEqual(mali_half_prefill.prepare('CL')['prepared_kernels'],2)

  def test_public_dispatch_and_function_boundary_preserve_bits_padding_and_rows(self):
    self.assertEqual(mali_half_prefill.prepare('CL')['prepared_kernels'],2)
    rng=np.random.default_rng(20261024)
    for columns in (2048,6144):
      source=nn.Linear(2048,columns,bias=False)
      source.weight=Tensor(rng.normal(0,.02,(columns,2048)).astype(np.float16),device='CL').realize()
      raw=MaliLinear(source,prepare_gemm=False,consistent_decode=True)
      bounded=MaliLinear(source,prepare_gemm=False,consistent_decode=True)
      bounded.bounded_half_prefill=True
      plain=TinyJit(lambda x,rows:raw(x[:,:rows]).realize())
      fast=TinyJit(lambda x,rows:bounded(x[:,:rows]).realize())
      with launches() as records:
        for index,count in enumerate((128,30,1,64,112,32,65,13,36,63,33,128)):
          values=rng.normal(0,(.3,.003,3.)[index%3],(1,128,2048)).astype(np.float16)
          values.reshape(-1)[:6]=np.array([0,2**-24,-2**-24,2**-14,1,-1],dtype=np.float16)
          values[:,count:]=np.nan
          x=Tensor(values,device='CL').realize()
          rows=UOp.variable('half_test_rows',1,128).bind(count)
          expected=plain(x,rows).pad_to((1,128,columns)).numpy().copy()
          actual=fast(x,rows).pad_to((1,128,columns)).numpy().copy()
          np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
          self.assertTrue(np.isfinite(actual).all())
          np.testing.assert_array_equal(actual[:,count:],0)
          self.assertTrue(records)
          self.assertEqual(records[-1],((columns//(64 if columns==2048 else 48),(count+31)//32,1),
            (8,16,1),{'half_test_rows':count}))

      def project(x,weight):
        result=mali_half_prefill.bounded_linear(x,weight)
        assert result is not None
        return result
      outer=function(project)
      compound=TinyJit(lambda x,start,end:outer(x[:,:(end.minimum(128)-start)],source.weight).realize())
      original=TinyJit(lambda x,start,end:raw(x[:,:(end.minimum(128)-start)]).realize())
      with launches() as records:
        for start,end in ((16,144),(5,35),(16,17),(0,128),(8,120),(4,17)):
          count=min(end,128)-start
          values=rng.normal(0,.3,(1,128,2048)).astype(np.float16)
          values[:,count:]=np.nan
          x=Tensor(values,device='CL').realize()
          args=(x,UOp.variable('half_start',0,16).bind(start),UOp.variable('half_end',17,144).bind(end))
          expected=original(*args).pad_to((1,128,columns)).numpy().copy()
          actual=compound(*args).pad_to((1,128,columns)).numpy().copy()
          np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
          np.testing.assert_array_equal(actual[:,count:],0)
          self.assertEqual(records[-1],((columns//(64 if columns==2048 else 48),(count+31)//32,1),
            (8,16,1),{'half_start':start,'half_end':end}))


if __name__=='__main__':unittest.main()
