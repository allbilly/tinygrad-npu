"""ABBA comparison of the retained GPT-2 CPU/NPU paths with measured clocks."""
import argparse, hashlib, json, os, statistics, subprocess, time
from pathlib import Path


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report-dir',type=Path,required=True)
  parser.add_argument('--reference',type=Path,required=True)
  parser.add_argument('--warmups',type=int,default=5)
  parser.add_argument('--runs',type=int,default=5)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  args.report_dir.mkdir(parents=True,exist_ok=False)
  files = [root/'examples/gpt2.py',*(root/'tinygrad').rglob('*rockchip*.py'),*(root/'tinygrad/llm/kernels').glob('rockchip*.c'),
           *(root/'extra/rockchip').glob('*.py')]
  hashes = {str(p.relative_to(root)):digest(p) for p in files}
  report = {'complete':False,'warmups':args.warmups,'runs':args.runs,'source_sha256':hashes,'blocks':[],
            'precision':'FP16 weights and NPU operands; FP32 CMAC accumulation',
            'control':'native loop and SIMD greedy; driver input sync and serial decode attention',
            'candidate':'native loop and SIMD greedy; probed input cache cleaning and parked CPU attention workers',
            'rkllm':'Installed 1.3.1 importer rejects GPT2LMHeadModel; no direct GPT-2 timing is available'}
  target = args.report_dir/'comparison.json'
  def save(): target.write_text(json.dumps(report,indent=2)+'\n')
  save()
  try:
    for index,mode in enumerate(['control','candidate','candidate','control']):
      result = args.report_dir/f'{index}-{mode}.json'
      log = result.with_suffix('.log')
      command = ['taskset','-c','4-7',str(root/'.venv/bin/python'),'-m','extra.rockchip.benchmark_gpt2',
                 '--lengths','8','32','128','--warmups',str(args.warmups),'--runs',str(args.runs),
                 '--npu','--native','--optimized','--bulk','--reference',str(args.reference),'--report',str(result)]
      if mode=='control': command += ['--no-cache-cleaner','--no-attention-pool']
      env = os.environ.copy()
      env.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
      block = {'index':index,'mode':mode,'command':command,'status':'running','clocks':[]}
      report['blocks'].append(block)
      save()
      print(json.dumps({'event':'start','index':index,'mode':mode}),flush=True)
      start = time.monotonic()
      with log.open('w') as output:
        process = subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
        while process.poll() is None:
          paths = [*Path('/sys/class/devfreq').glob('*/cur_freq'),*Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')]
          block['clocks'].append({'time':time.monotonic(),'frequencies':{str(p):int(p.read_text()) for p in paths}})
          time.sleep(.2)
        code = process.wait()
      block.update(exit_code=code,seconds=time.monotonic()-start)
      if code: raise RuntimeError(f'{mode} failed; see {log}')
      data = json.loads(result.read_text())
      if not data['complete']: raise AssertionError(f'{mode} did not complete')
      for case in data['cases']:
        trials = case['trials']
        if len(trials)!=args.warmups+args.runs or any(len(r['generated_ids'])!=32 for r in trials):
          raise AssertionError('Incomplete warmup or generated sequence')
        for row in trials:
          if row['generated_ids']!=case['generated_ids'] or row['decode_submits']!=31*51 or row['decode_tasks']!=31*153:
            raise AssertionError('Token sequence or NPU execution count differs')
      block.update(status='complete',result=data)
      if any(digest(root/name)!=value for name,value in hashes.items()): raise AssertionError('Measured source changed')
      save()
      print(json.dumps({'event':'finished','index':index,'mode':mode,
                       'rates':[(c['input_tokens'],c['prefill_tps'],c['decode_tps']) for c in data['cases']]}),flush=True)
    report['summary'] = []
    for length in (8,32,128):
      values = {}
      for mode in ('control','candidate'):
        cases = [next(c for c in b['result']['cases'] if c['input_tokens']==length) for b in report['blocks'] if b['mode']==mode]
        if any(c['generated_ids']!=cases[0]['generated_ids'] for c in cases): raise AssertionError('Outputs vary across processes')
        values[mode] = {name:statistics.median(c[name] for c in cases) for name in ('prefill_tps','decode_tps')}
      report['summary'].append({'input_tokens':length,**values,
        'decode_speedup':values['candidate']['decode_tps']/values['control']['decode_tps']})
    report['complete'] = True
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally: save()


if __name__=='__main__': main()
