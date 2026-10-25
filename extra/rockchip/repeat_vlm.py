"""Run the reversed half of the matched prepared-vision comparison, recording clocks."""
import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--output-dir',type=Path)
  parser.add_argument('--full',action='store_true',help='Run both halves of the matched engine-order comparison')
  parser.add_argument('--vendor-model',type=Path,help='Level-0 reference with its checksum/build provenance')
  parser.add_argument('--vendor-binary',type=Path)
  parser.add_argument('--tokens',type=int,help='One output length in balanced onpu/vendor/vendor/onpu order')
  parser.add_argument('--vision-gpu',action=argparse.BooleanOptionalAction,default=True)
  parser.add_argument('--quiet-seconds',type=float,default=60)
  args = parser.parse_args()
  if args.tokens is not None and (args.tokens<2 or args.tokens>128 or args.full):parser.error('Use2..128 tokens without --full')
  repo = Path(__file__).resolve().parents[2]
  root = args.root.resolve()
  from extra.rockchip.rkllm_reference import validate_reference
  vendor_model = args.vendor_model or root/'vendor/text-w8a8-opt0-ctx512.rkllm'
  vendor_metadata = validate_reference(vendor_model,context=512)
  vendor_binary = args.vendor_binary or root/'vendor/benchmark_vlm_rkllm'
  output = args.output_dir.resolve() if args.output_dir else root/'prepared-repeats'
  output.mkdir(exist_ok=True)
  progress = output/'progress.json'
  if progress.exists():raise ValueError('Repeat evidence directory is already in use')
  source_paths = [p for p in (repo/'tinygrad').rglob('*') if p.suffix in ('.py','.c','.h')]
  source_paths += [Path(__file__),repo/'extra/rockchip/benchmark_vlm.py',repo/'extra/rockchip/benchmark_vlm_rkllm.cpp']
  def digest(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
  hashes = {str(p):digest(p) for p in source_paths}
  report = {'complete':False,'source_sha256':hashes,'reference':vendor_metadata,'vendor_binary_sha256':digest(vendor_binary),
            'vision_gpu':args.vision_gpu,'fixture_sha256':{name:digest(root/'demo'/name)
              for name in ('pixels.npy','input_ids.json','input_embeddings.f32')},'runs':[]}
  phase = {'engine':None,'tokens':None,'measured':False}
  stop = threading.Event()
  paths = [Path('/sys/devices/system/cpu')/f'cpu{cpu}/cpufreq/scaling_cur_freq' for cpu in (4,6)]
  paths += [p/'cur_freq' for p in Path('/sys/class/devfreq').iterdir()]
  paths += [Path('/sys/class/thermal/thermal_zone0/temp')]
  def sampler():
    with (output/'clocks.jsonl').open('w') as log:
      while not stop.is_set():
        values = {}
        for path in paths:
          try:values[str(path)] = path.read_text().strip()
          except OSError:pass
        log.write(json.dumps({'time':time.time(),**phase,'values':values})+'\n')
        log.flush()
        stop.wait(1)
  thread = threading.Thread(target=sampler)
  if hasattr(os,'sched_getaffinity'):
    little = set(os.sched_getaffinity(0)).intersection({0,1,2,3})
    if little:os.sched_setaffinity(0,little)
  thread.start()
  try:
    plan = [('onpu',128,'b'),('vendor',128,'b'),('vendor',32,'b'),('onpu',32,'b')]
    if args.full:plan=[('onpu',32,'a'),('vendor',32,'a'),('vendor',128,'a'),('onpu',128,'a')]+plan
    if args.tokens is not None:plan=[('onpu',args.tokens,'a'),('vendor',args.tokens,'a'),('vendor',args.tokens,'b'),('onpu',args.tokens,'b')]
    for engine,tokens,half in plan:
      label = f'{engine}-{tokens}-{half}'
      if engine=='onpu':
        command = ['/usr/bin/taskset','-c','4-7',sys.executable,str(repo/'extra/rockchip/benchmark_vlm.py'),
          '--checkpoint',str(root/'hf'),'--fixture',str(root/'demo'),'--vision-npu','on','--staged-vision',
          '--staged-attention','ROCKCHIP','--npu-connector','--fused-vision-mlp',
          '--native-generation-loop','--reset-before-vision','--cpu-vision-attention','--text-npu','on','--precision','int8',
          '--tokens',str(tokens),'--runs','3','--report',str(output/(label+'.json'))]
        if args.vision_gpu:command+=['--shared-gpu-softmax','--prepared-vision-mlp']
      else:
        command = ['flock','/tmp/rk3588_npu_submit.lock',sys.executable,str(repo/'extra/rockchip/sdk_entry.py'),
          '/usr/bin/taskset','-c','4-7',str(vendor_binary),str(root/'vendor/vision.rknn'),
          str(vendor_model),str(root/'demo'),str(tokens),'3','3']
      phase.update(engine=engine,tokens=tokens,measured=False)
      print(json.dumps({'starting':label,'quiet_seconds':args.quiet_seconds}),flush=True)
      time.sleep(args.quiet_seconds)
      started = time.time()
      with (output/(label+'.log')).open('w') as log:
        process = subprocess.Popen(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        assert process.stdout is not None
        for line in process.stdout:
          log.write(line)
          if not line.startswith('{'):continue
          try:item=json.loads(line)
          except ValueError:continue
          if item.get('event')=='trial':phase['measured']=not item['warmup'] or item.get('index')==2
        result = process.wait()
      report['runs'].append({'label':label,'command':command,'started':started,'ended':time.time(),'exit_code':result})
      progress.write_text(json.dumps(report,indent=2)+'\n')
      if result:raise RuntimeError(f'{label} failed with {result}')
      if engine=='vendor' and 'model_dtype: W8A8' not in (output/(label+'.log')).read_text():
        raise ValueError('Vendor runtime did not confirm W8A8')
      print(json.dumps({'completed':label}),flush=True)
    if any(digest(Path(p))!=value for p,value in hashes.items()):raise RuntimeError('Sources changed during repeats')
    if any(digest(root/'demo'/name)!=value for name,value in report['fixture_sha256'].items()):
      raise RuntimeError('Matched fixture changed during repeats')
    report['complete']=True
    progress.write_text(json.dumps(report,indent=2)+'\n')
  finally:
    stop.set()
    thread.join()


if __name__=='__main__':main()
