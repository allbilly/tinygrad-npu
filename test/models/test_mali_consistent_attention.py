"""Bit-exact Mali attention consistency across cached tile boundaries."""
import os, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestMaliConsistentGatedAttention(unittest.TestCase):
  def test_consistent_decode_batch_rows_match_serial_across_tiles(self):
    """Small GQA geometries exercise 1/2/4 GQA and head dims 16/32."""
    from tinygrad.llm.kernels.mali_attention import SharedAttention, attention

    rng=np.random.default_rng(61024)
    context,capacity,kv_heads=1100,8,2
    # (head dimension, query heads): GQA ratios are 1, 2 and 4.
    geometries=((16,2),(32,4),(32,8))
    # Each captured batch invocation binds a different row count and sequence end.
    # The row block straddles tile 512 or 1024, including its first and last row.
    cases=((2,513),(3,1026),(8,1031))

    for dim,heads in geometries:
      gqa=heads//kv_heads
      batched=TinyJit(lambda q,kv,end,rows:
        attention(q[:,:,:rows],kv,end,subgroups=True,consistent_decode=True).realize())
      batched_no_subgroups=TinyJit(lambda q,kv,end,rows:
        attention(q[:,:,:rows],kv,end,subgroups=False,consistent_decode=True).realize())
      single=TinyJit(lambda q,kv,end:
        attention(q,kv,end,subgroups=True,consistent_decode=True).realize())
      shared=SharedAttention(subgroups=True,consistent_decode=True)
      shared.prepare(heads,dim,context,capacity,'CL')
      shared_run=TinyJit(lambda q,kv,end,rows:shared(q[:,:,:rows],kv,end).realize())

      for rows,end in cases:
        self.assertLess(end,context)
        query_data=rng.normal(0,.3,(1,heads,capacity,dim)).astype(np.float32)
        cache_data=rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
        # Finite poison beyond the current request catches a missing end mask;
        # later rows inside the request remain non-poisoned and catch a missing
        # lower-right causal mask (each query row has its own visible KV length).
        cache_data[0,0,:,end:,:]=np.float16(32)
        cache_data[1,0,:,end:,:]=np.float16(-64)
        self.assertTrue(np.isfinite(cache_data).all())
        q=Tensor(query_data,device='CL').realize()
        kv=Tensor(cache_data,device='CL').realize()
        end_arg=UOp.variable('end',1,context).bind(end)
        rows_arg=UOp.variable('rows',1,capacity).bind(rows)

        batch_out=batched(q,kv,end_arg,rows_arg)
        batch=batch_out.pad_to((1,heads,capacity,dim)).numpy()[:,:,:rows].copy()
        batch_bits=batch.view(np.uint32).copy()
        # The opt-in must force the same reduction path even when subgroup use
        # is requested. The patch explicitly disables subgroups at capacity<=8.
        no_subgroups_out=batched_no_subgroups(q,kv,UOp.variable('end',1,context).bind(end),
                                             UOp.variable('rows',1,capacity).bind(rows))
        no_subgroups=no_subgroups_out.pad_to((1,heads,capacity,dim)).numpy()[:,:,:rows].copy()
        np.testing.assert_array_equal(batch_bits,no_subgroups.view(np.uint32))
        self.assertIsNot(batch_out.uop,no_subgroups_out.uop)
        shared_out=shared_run(q,kv,UOp.variable('end',1,context).bind(end),
                              UOp.variable('rows',1,capacity).bind(rows))
        shared_values=shared_out.pad_to((1,heads,capacity,dim)).numpy()[:,:,:rows].copy()
        np.testing.assert_array_equal(batch_bits,shared_values.view(np.uint32))
        self.assertIsNot(batch_out.uop,shared_out.uop)

        serial=[]
        reference=np.empty((1,heads,rows,dim),dtype=np.float32)
        for row in range(rows):
          row_query=Tensor(query_data[:,:,row:row+1,:],device='CL').realize()
          visible=end-rows+row+1
          row_end=UOp.variable('end',1,context).bind(visible)
          serial_out=single(row_query,kv,row_end)
          self.assertIsNot(batch_out.uop,serial_out.uop)
          serial.append(serial_out.numpy().copy())

          for head in range(heads):
            kv_head=head//gqa
            keys=cache_data[0,0,kv_head,:visible].astype(np.float32)
            values=cache_data[1,0,kv_head,:visible].astype(np.float32)
            scores=(keys@query_data[0,head,row])/np.float32(np.sqrt(dim))
            probabilities=np.exp(scores-scores.max())
            probabilities/=probabilities.sum()
            reference[0,head,row]=probabilities@values

        serial_values=np.concatenate(serial,axis=2)
        serial_bits=serial_values.view(np.uint32)
        np.testing.assert_array_equal(batch_bits,serial_bits)
        # Existing Mali attention tests use these tolerances for the independent
        # float32 reference. This is a math sanity check, not a relaxed parity gate.
        np.testing.assert_allclose(batch,reference,atol=4e-6,rtol=2e-4)
        # Ensure subsequent TinyJit invocations did not overwrite this result.
        np.testing.assert_array_equal(batch_out.pad_to((1,heads,capacity,dim)).numpy()[:,:,:rows].view(np.uint32),batch_bits)


if __name__=='__main__':unittest.main()
