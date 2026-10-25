"""Check layout-preserving CPU SwiGLU loop orders; never changes production kernels."""
import argparse, functools, json, os, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.llm.kernels import rockchip_cpu as cpu
from tinygrad.uop.ops import Ops

VECTOR_SIGMOID = cpu._vector_sigmoid_source()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  factory = cpu._swiglu_pack
  rng = np.random.default_rng(26)
  rows,depth,input_rows = 128,3072,64
  data = rng.normal(0,3,(rows,depth*2)).astype(np.float32)
  data.flat[:6] = [-0.,0.,-80.,80.,-1.,1.]
  source = np.concatenate([data[s:s+input_rows].reshape(input_rows,depth//2,4).transpose(1,0,2).flatten()
                           for s in range(0,rows,input_rows)])
  x = Tensor(source,device='CPU').realize()
  report = {'cpu_affinity':sorted(os.sched_getaffinity(0)),'rows':rows,'depth':depth,'input_rows':input_rows,'results':[]}
  try:
    for tile in [21,64]:
      golden = None
      for width in [0,4,8,16,-4]:
        @functools.cache
        def variant(*inputs):
          program = factory(*inputs,vectorized=False)
          text = next(u.arg for u in program.src if u.op is Ops.SOURCE)
          if width:
            block_width = abs(width)
            text = text.replace(f'for(int row=0;row<{rows};row++)for(int c=begin;c<end;c++) {{',
              f'for(int block=begin;block<end;block+={block_width})for(int row=0;row<{rows};row++)'
              f'for(int c=block;c<block+{block_width};c+={4 if width<0 else 1}) {{')
          if width<0:
            text = VECTOR_SIGMOID+text
            text = text.replace('float gate=src[g],up=src[u];',
                                'float4 gate=*(const float4*)(src+g),up=*(const float4*)(src+u);')
            text = text.replace('float reciprocal;rockchip_sigmoid(&reciprocal,&gate);','float4 reciprocal=rockchip_sigmoid_vector(gate);')
            text = text.replace('float activated=gate*reciprocal,value=activated*up;','float4 activated=gate*reciprocal,value=activated*up;')
            text = text.replace(f'out[start*{depth}+(c/8*m+row-start)*8+c%8]=(__fp16)value;',
                                f'*(half4*)(out+start*{depth}+(c/8*m+row-start)*8+c%8)=__builtin_convertvector(value,half4);')
          return program.replace(src=tuple(u.replace(arg=text) if u.op is Ops.SOURCE else
            u.replace(arg=cpu._renderer().compiler.compile_cached(text)) if u.op is Ops.BINARY else u for u in program.src))
        cpu._swiglu_pack = variant
        run = TinyJit(lambda a:cpu.swiglu_pack(a,depth,tile,input_rows,split_k=tile==64).realize())
        times = []
        for i in range(17):
          start = time.perf_counter()
          result = run(x)
          elapsed = (time.perf_counter()-start)*1000
          if i>=8: times.append(elapsed)
        output = result.numpy().view(np.uint16)
        if golden is None: golden = output
        np.testing.assert_array_equal(output,golden)
        report['results'].append({'tile_rows':tile,'channel_block':width,'bitwise_equal':True,
                                  'median_ms':statistics.median(times),'all_ms':times})
        args.report.write_text(json.dumps(report,indent=2)+'\n')
  finally: cpu._swiglu_pack = factory


if __name__=='__main__': main()
