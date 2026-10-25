"""Repeated 128-input/64-output RK3588 comparison with a verified level-0 reference.

Published clocks are not imposed on the host. Both engines use the current host
policy; raw timings without matching quality evidence cannot become verified wins.
Production workers must be unloaded before this exclusive hardware benchmark.
Each engine must reproduce its own greedy token stream across warmups and both
ABBA cohorts; quantization can make the two engines' streams differ.
"""
from __future__ import annotations
import argparse, json, math, os, signal, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path
from extra.rockchip.rkllm_reference import file_hash, validate_reference


def summarize(candidate:list[dict], reference:list[dict], *, context:int, quality_passed:bool=False) -> dict:
  measured = {}
  for name,rows in [('onpu',candidate),('rkllm',reference)]:
    if sum(row.get('warmup') is True for row in rows)<3:raise ValueError('Need at least three warmups per engine')
    values = [row for row in rows if row.get('warmup') is False]
    if len(values)<3:raise ValueError('Need at least three measured runs per engine')
    for row in rows:
      if row.get('context',context)!=context or row.get('prompt_tokens')!=128 or len(row.get('generated_ids',[]))!=64:
        raise ValueError('Different context, prompt length or generated count')
      if row['generated_ids']!=values[0]['generated_ids']:
        raise ValueError(f'{name}: greedy output changed across warmups, measured runs or ABBA cohorts')
      if name=='rkllm' and (row.get('sdk_prefill_tokens')!=128 or row.get('kv_tokens')!=191):
        raise ValueError('SDK did not retain the full requested prefix/output history')
    for row in values:
      if any(not math.isfinite(row[key]) or row[key]<=0 for key in ('first_token_ms','decode_tps')):raise ValueError('Invalid timing')
    measured[name] = {'runs':len(values),'median_first_token_ms':statistics.median(row['first_token_ms'] for row in values),
                      'median_prefill_proxy_tps':statistics.median(128000/row['first_token_ms'] for row in values),
                      'median_decode_tps':statistics.median(row['decode_tps'] for row in values),
                      'prefill_range_tps':[min(128000/row['first_token_ms'] for row in values),max(128000/row['first_token_ms'] for row in values)],
                      'decode_range_tps':[min(row['decode_tps'] for row in values),max(row['decode_tps'] for row in values)]}
  a,b = measured['onpu'],measured['rkllm']
  ratios = {'prefill':a['median_prefill_proxy_tps']/b['median_prefill_proxy_tps'],'decode':a['median_decode_tps']/b['median_decode_tps']}
  separated = a['prefill_range_tps'][0]>b['prefill_range_tps'][1] and a['decode_range_tps'][0]>b['decode_range_tps'][1]
  faster = all(value>1 for value in ratios.values())
  outcome = 'measured_loss' if not faster else 'quality_pending' if not quality_passed else 'repeat_needed' if not separated else 'verified_win'
  return {'engines':measured,'speed_ratios':ratios,'observed_medians_faster_both':faster,'ranges_separated_both':separated,
          'quality_passed':quality_passed,'qualified_win':outcome=='verified_win','outcome':outcome}


def source_snapshot(root:Path) -> dict:
  paths = [p for p in (root/'tinygrad').rglob('*') if p.is_file() and p.suffix in ('.py','.c','.h')]
  paths += [Path(__file__),root/'extra/rockchip/benchmark_inference.py',root/'extra/rockchip/rkllm_reference.py',
            root/'extra/rockchip/calibrate_text_int8.py']
  return {str(path.relative_to(root)):file_hash(path) for path in paths}


def check_quality(path:Path, manifest:Path, checkpoint_hash:str, reference_hash:str, context:int, args, snapshot:dict) -> dict:
  data,requests = json.loads(path.read_text()),json.loads(manifest.read_text())
  if not data.get('complete') or not data.get('passed') or data.get('positions',0)<256 or data.get('max_perplexity_ratio',100)>1.02:
    raise ValueError('Comparable-quality evidence is incomplete or uses a weaker gate')
  ratio = data.get('perplexity_ratio_vs_rkllm',float('inf'))
  if not math.isfinite(ratio) or not 0<ratio<=data['max_perplexity_ratio']:raise ValueError('Quality ratio contradicts the passed flag')
  if data.get('manifest_sha256')!=file_hash(manifest) or requests.get('context')!=context:
    raise ValueError('Quality requests/context differ')
  stock,candidate = data['provenance']
  if stock.get('model_sha256')!=reference_hash or candidate.get('checkpoint_sha256')!=checkpoint_hash:
    raise ValueError('Quality weights differ from the performance comparison')
  scale_path = getattr(args,'input_scales',None)
  if candidate.get('input_scales_sha256')!=(file_hash(scale_path) if scale_path else None):
    raise ValueError('Quality input scaling differs')
  if candidate.get('device')!=args.device or candidate.get('gpu_fuse_projections',False)!=args.gpu_fuse_projections or \
     candidate.get('gpu_q8_head',False)!=args.gpu_q8_head or \
     candidate.get('gpu_q8_projections',False)!=getattr(args,'gpu_q8_projections',False):
    raise ValueError('Quality device or GPU configuration differs')
  gpu = candidate.get('gpu_kernels') or {}
  if bool(gpu.get('q8_projection_layers'))!=getattr(args,'gpu_q8_projections',False):
    raise ValueError('Quality native Q8 body projections did not activate in the requested configuration')
  if getattr(args,'gpu_q8_projections',False):
    if gpu.get('fused_qkv') or gpu.get('fused_gate_up') or candidate.get('gpu_npu',False):
      raise ValueError('Quality native Q8 body projections did not activate in the requested configuration')
  if args.device=='CPU':
    placement = candidate.get('placement') or {}
    if args.prefill!='on' or args.decode!='on' or candidate.get('precision')!='int8' or placement.get('precision')!='int8' or \
       placement.get('prefill_layers')!=placement.get('layers') or placement.get('decode_layers')!=placement.get('layers') or \
       not all(placement.get(key) for key in ('layers','cpu_attention_blocks','native_replay','fused_ffn_blocks','cpu_norms',
                                             'cpu_pool','cpu_fast_softmax','cache_cleaning','wake_helper')):
      raise ValueError('CPU+NPU quality configuration differs')
  elif args.prefill!='off' or args.decode!='off' or candidate.get('gpu_npu',False):
    raise ValueError('GPU quality adapter currently certifies assistance Off')
  hashes = candidate.get('source_sha256',{})
  if any(hashes.get(name)!=value for name,value in snapshot.items() if name.startswith('tinygrad/')):
    raise ValueError('Quality numerical source differs')
  return {'path':str(path.resolve()),'sha256':file_hash(path),'manifest_sha256':file_hash(manifest),
          'perplexity_ratio_vs_rkllm':data['perplexity_ratio_vs_rkllm'],'passed':True}


def run(command:list[str], log:Path, timeout:float):
  with log.open('w') as output:
    process = subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    try:code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      os.killpg(process.pid,signal.SIGKILL)
      process.wait()
      raise
  if code:raise RuntimeError(f'Benchmark exited {code}; see {log}')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('checkpoint','reference','input','sdk-binary','output'):
    parser.add_argument('--'+name,type=Path,required=True)
  parser.add_argument('--context',type=int,default=512)
  parser.add_argument('--runs',type=int,default=3)
  parser.add_argument('--warmups',type=int,default=3)
  parser.add_argument('--device',choices=('CL','CPU'),default='CL')
  parser.add_argument('--prefill',choices=('off','auto','on'),default='off')
  parser.add_argument('--decode',choices=('off','auto','on'),default='off')
  parser.add_argument('--gpu-fuse-projections',action='store_true')
  parser.add_argument('--gpu-q8-head',action='store_true')
  parser.add_argument('--gpu-q8-projections',action='store_true')
  parser.add_argument('--input-scales',type=Path)
  parser.add_argument('--quality-report',type=Path)
  parser.add_argument('--quality-manifest',type=Path)
  parser.add_argument('--timeout',type=float,default=1800)
  parser.add_argument('--quiet-seconds',type=float,default=60)
  parser.add_argument('--pause-converter',action='append',default=[])
  args = parser.parse_args()
  if args.runs<3 or args.warmups<3:parser.error('Qualification needs at least three warmups and three measurements')
  if args.device=='CPU' and (args.prefill!='on' or args.decode!='on' or args.gpu_fuse_projections or args.gpu_q8_head):
    parser.error('CPU comparison uses the prepared INT8 NPU profile with both phases On')
  if args.input_scales and args.device!='CPU':parser.error('Input scaling requires CPU/NPU INT8')
  if args.gpu_q8_projections and (args.device!='CL' or args.prefill!='off' or args.decode!='off' or args.gpu_fuse_projections):
    parser.error('GPU Q8 body requires GPU-only inference without projection fusion')
  ids = [int(x) for x in args.input.read_text().split()]
  if len(ids)!=128 or args.context<192:parser.error('Expected128 input positions androom for64 output positions')
  reference = validate_reference(args.reference,context=args.context)
  root = Path(__file__).resolve().parents[2]
  snapshot = source_snapshot(root)
  checkpoint_hash = file_hash(args.checkpoint)
  quality = None
  if args.quality_report:
    if args.quality_manifest is None:parser.error('Quality report needs its actual manifest')
    quality = check_quality(args.quality_report,args.quality_manifest,checkpoint_hash,reference['artifact_sha256'],args.context,args,snapshot)
  opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open('http://127.0.0.1:8091/api/status',timeout=10) as response:status = json.load(response)
  if status['runtime']['sessions']:raise RuntimeError('Unload production workers before exclusive hardware measurements')
  args.output.mkdir(parents=True,exist_ok=False)
  target = args.output/'comparison.json'
  report = {'complete':False,'protocol':{'input_tokens':128,'output_tokens':64,'context':args.context,
            'prefill_metric':'128 / time-to-first-token; includes first-token overhead','clock_policy':'unchanged host policy'},
            'source_sha256':snapshot,'checkpoint_sha256':checkpoint_hash,'reference':reference,'quality':quality,
            'candidate':{'checkpoint':str(args.checkpoint.resolve()),'device':args.device,'prefill':args.prefill,'decode':args.decode,
                         'gpu_fuse_projections':args.gpu_fuse_projections,'gpu_q8_head':args.gpu_q8_head,
                         'gpu_q8_projections':args.gpu_q8_projections,
                         'input_scales_sha256':file_hash(args.input_scales) if args.input_scales else None},
            'input_ids':ids,'input_sha256':file_hash(args.input),'sdk_binary_sha256':file_hash(args.sdk_binary),'runs':[]}
  paused,rows = [],{'onpu':[],'rkllm':[]}
  phase,stop = {'engine':None,'cohort':None},threading.Event()
  def sample():
    paths = [*Path('/sys/class/devfreq').glob('*/cur_freq'),
             *Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'),Path('/sys/class/thermal/thermal_zone0/temp')]
    with (args.output/'clocks.jsonl').open('w') as log:
      while not stop.is_set():
        values = {}
        for path in paths:
          try:values[str(path)]=path.read_text().strip()
          except OSError:pass
        log.write(json.dumps({'time':time.time(),**phase,'values':values})+'\n')
        log.flush()
        stop.wait(1)
  sampler = threading.Thread(target=sample)
  def save():target.write_text(json.dumps(report,indent=2)+'\n')
  save()
  sampler.start()
  try:
    for name in args.pause_converter:
      if not name.startswith('onpu-parity-'):raise ValueError('Can pause only owned parity converters')
      state = json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',name],text=True))
      if state['Running'] and not state['Paused']:
        subprocess.run(['docker','pause',name],check=True,stdout=subprocess.DEVNULL)
        paused.append(name)
    for index,engine in enumerate(('rkllm','onpu','onpu','rkllm')):
      label = f'{index}-{engine}'
      phase.update(engine=engine,cohort=label)
      print(json.dumps({'starting':label,'quiet_seconds':args.quiet_seconds}),flush=True)
      time.sleep(args.quiet_seconds)
      if engine=='rkllm':
        command = ['flock','/tmp/rk3588_npu_submit.lock','taskset','-c','4-7',sys.executable,'-m','extra.rockchip.sdk_entry',
                   str(args.sdk_binary),str(args.reference),str(args.input),str(args.context),'64',str(args.warmups),str(args.runs)]
      else:
        command = ['taskset','-c','4-7',sys.executable,'-m','extra.rockchip.benchmark_inference','--checkpoint',str(args.checkpoint),
                   '--input',str(args.input),'--device',args.device,'--prefill',args.prefill,'--decode',args.decode,
                   '--context',str(args.context),'--tokens','64','--warmups',str(args.warmups),'--runs',str(args.runs),
                   '--report',str(args.output/(label+'.json'))]
        if args.gpu_fuse_projections:command+=['--gpu-fuse-projections']
        if args.gpu_q8_head:command+=['--gpu-q8-head']
        if args.gpu_q8_projections:command+=['--gpu-q8-projections']
        if args.input_scales:command+=['--input-scales',str(args.input_scales)]
        if args.device=='CPU':
          command+=['--precision','int8','--tile-columns','24576','--chunk-size','128','--fuse-projections',
                    '--cpu-attention','--native-replay','--fused-ffn','--fused-qkv','--cpu-norm','--split-down',
                    '--cpu-pool','--cpu-fast-softmax','--cache-cleaning','--wake-helper']
      started = time.time()
      run(command,args.output/(label+'.log'),args.timeout)
      if engine=='rkllm':
        raw = (args.output/(label+'.log')).read_text()
        if 'model_dtype: W8A8' not in raw:raise ValueError('Reference runtime did not confirm W8A8')
        trials = []
        for line in raw.splitlines():
          try:item=json.loads(line)
          except ValueError:continue
          if isinstance(item,dict) and item.get('event')=='run':trials.append(item)
      else:
        candidate = json.loads((args.output/(label+'.json')).read_text())
        if not candidate['complete'] or candidate['input_ids']!=ids or candidate['context']!=args.context:
          raise ValueError('Candidate request mismatch')
        gpu = candidate.get('gpu_kernels') or {}
        if args.gpu_q8_head and not gpu.get('q8_output_head'):raise ValueError('Requested Q8 head did not activate')
        if args.gpu_fuse_projections and not (gpu.get('fused_qkv') or gpu.get('fused_gate_up')):
          raise ValueError('Requested GPU joint projections did not activate')
        if args.gpu_q8_projections and not gpu.get('q8_projection_layers'):
          raise ValueError('Requested native Q8 body projections did not activate')
        trials = candidate['trials']
      rows[engine] += trials
      report['runs'].append({'label':label,'command':command,'started':started,'ended':time.time(),'trials':trials})
      save()
    if source_snapshot(root)!=snapshot:raise RuntimeError('Sources changed during the matched cohort')
    report.update(summary=summarize(rows['onpu'],rows['rkllm'],context=args.context,quality_passed=quality is not None),complete=True)
  except BaseException as exc:
    report['error']=f'{type(exc).__name__}: {exc}'
    raise
  finally:
    stop.set()
    sampler.join()
    for name in paused:subprocess.run(['docker','unpause',name],check=False,stdout=subprocess.DEVNULL)
    save()
  print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
