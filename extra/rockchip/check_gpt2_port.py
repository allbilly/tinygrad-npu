"""Run CPU and ported GPT-2 while rejecting reference-folder and archive dependencies."""
import argparse, json, os, sys
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--tokens',type=int,default=64)
  args = parser.parse_args()
  forbidden = ['/home/orangepi/tinygrad-1500','/home/orangepi/tinygrad-1500-clean','/home/orangepi/archives/']
  opened:set[str] = set()
  def audit(event, values):
    paths = []
    if event in ('open','ctypes.dlopen') and values and isinstance(values[0],str): paths.append(values[0])
    if event=='subprocess.Popen':
      paths.extend(str(v) for v in values[1] if isinstance(v,(str,Path)))
    for path in paths:
      canonical = str(Path(path).resolve()) if event in ('open','ctypes.dlopen') else path
      if any(prefix in canonical or prefix in path for prefix in forbidden): raise RuntimeError(f'Unported dependency accessed: {path}')
      if event in ('open','ctypes.dlopen'): opened.add(canonical)
  sys.addaudithook(audit)
  os.environ.update(DEV='CPU',HALF='1',MAX_CONTEXT='128',JIT_BATCH_SIZE='0')
  from examples.gpt2 import GPT2
  from tinygrad import Device
  from tinygrad.llm.kernels.rockchip_gpt2 import configure_gpt2
  gpt = GPT2.build('gpt2')
  prompt = 'The quick brown fox jumps over the lazy'
  expected = gpt.generate(prompt,args.tokens,0.0)[0]
  placement = configure_gpt2(gpt.model)
  actual = gpt.generate(prompt,args.tokens,0.0)[0]
  loop = gpt.model.rockchip_gpt2.get('loop')
  report = {'complete':actual==expected,'tokens':args.tokens,'matches_cpu':actual==expected,'text':actual,
            'placement':placement,'native_loop_tokens':getattr(loop,'tokens',0),
            'npu_submits':Device['ROCKCHIP'].submit_count,'npu_tasks':Device['ROCKCHIP'].task_count,
            'forbidden_paths':forbidden,'files_observed':len(opened),
            'reference_accesses':[p for p in sorted(opened) if any(prefix in p for prefix in forbidden)]}
  args.report.parent.mkdir(parents=True,exist_ok=True)
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)
  if not report['complete'] or not report['native_loop_tokens'] or report['reference_accesses']:
    raise AssertionError('GPT-2 port failed the independent long-generation check')


if __name__=='__main__': main()
