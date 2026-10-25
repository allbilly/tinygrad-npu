"""Compare CPU and shared-buffer GPU softmax within the actual NPU attention path."""
import argparse, json, os, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.kernels.rockchip_attention import NPUVisionAttention

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  rng = np.random.default_rng(44)
  report = {'complete':False,'cpu_affinity':sorted(os.sched_getaffinity(0)),'results':[]}
  cpu = NPUVisionAttention(12,reset_context=False,profile=True)
  gpu = None
  try:
    gpu = NPUVisionAttention(12,reset_context=False,profile=True,gpu_softmax=True)
    print(json.dumps({'event':'dma_buf_imported','bytes':gpu.scores.size+gpu.probabilities.size}),flush=True)
    for scale in (.3,1.5):
      values = [rng.normal(0,scale,(1,12,1024,64)).astype(np.float32) for _ in range(3)]
      inputs = [Tensor(value,device='CPU').realize() for value in values]
      golden = cpu(*inputs).numpy().copy()
      for label,operator in [('cpu',cpu),('shared_gpu',gpu)]:
        samples = []
        for index in range(9):
          start = time.perf_counter()
          actual = operator(*inputs).numpy()
          elapsed = (time.perf_counter()-start)*1000
          if index>=4:samples.append(elapsed)
        np.testing.assert_allclose(actual,golden,atol=1e-4,rtol=3e-3)
        # New Q/K/V contents must replace the previous image's data on both devices.
        changed = [Tensor(value*.7,device='CPU').realize() for value in values]
        output = operator(*changed).numpy().copy()
        expected = cpu(*changed).numpy().copy()
        np.testing.assert_allclose(output,expected,atol=1e-4,rtol=3e-3)
        assert float(abs(output-actual).max())>1e-3
        row = {'method':label,'scale':scale,'median_ms':statistics.median(samples),'all_ms':samples,
               'phases_ms':operator.phase_timings(),'max_error':float(abs(actual-golden).max())}
        report['results'].append(row)
        print(json.dumps(row),flush=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    report['complete'] = True
    args.report.write_text(json.dumps(report,indent=2)+'\n')
  finally:
    if gpu is not None: gpu.close()
    cpu.close()


if __name__=='__main__': main()
