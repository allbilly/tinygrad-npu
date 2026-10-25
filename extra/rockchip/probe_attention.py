"""Compare bounded CPU attention register tiles with exact scalar-path outputs."""
import argparse, functools, hashlib, json, os, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels import rockchip_cpu as cpu


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  factory = cpu._attention
  rng = np.random.default_rng(36)
  report = {'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'source_sha256':hashlib.sha256(Path(cpu.__file__).read_bytes()).hexdigest(),'results':[]}
  try:
    for capacity in [128,256]:
      q = Tensor(rng.normal(0,.8,(1,16,capacity,128)).astype(np.float32),device='CPU').realize()
      c = Tensor(rng.normal(0,.8,(2,1,8,512,128)).astype(np.float16),device='CPU').realize()
      golden = None
      for block,tile in [(0,0),(4,1),(4,2),(4,4),(8,1),(8,2),(8,4),(16,1),(16,2)]:
        cpu._attention = functools.partial(factory,blocked_rows=bool(block),query_rows=block or 4,channel_tile=tile)
        run = TinyJit(lambda q,c,l:cpu.prefill_attention(q,c,l).realize())
        timings = []
        for i in range(17):
          start = time.perf_counter()
          output = run(q,c,UOp.variable('length',1,512).bind(capacity))
          elapsed = (time.perf_counter()-start)*1000
          if i>=8: timings.append(elapsed)
        values = output.numpy()
        if golden is None: golden = values
        np.testing.assert_array_equal(values.view(np.uint32),golden.view(np.uint32))
        report['results'].append({'rows':capacity,'query_block':block,'channel_tile':tile,'bitwise_equal':True,
                                  'median_ms':statistics.median(timings),'all_ms':timings})
        args.report.write_text(json.dumps(report,indent=2)+'\n')
  finally: cpu._attention = factory


if __name__=='__main__': main()
