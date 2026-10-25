"""Paired FP16 Qwen3 comparison with in-place source guards and sampled clocks."""
import argparse, hashlib, json, os, statistics, subprocess, time
from pathlib import Path


def digest(path):
  with path.open('rb') as source: return hashlib.file_digest(source,'sha256').hexdigest()


def clock_sample():
  paths = [*Path('/sys/class/devfreq').glob('*/cur_freq'),
           *Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')]
  return {'time':time.time(),'frequencies':{str(p):int(p.read_text()) for p in paths}}


def stock_rows(path):
  rows = []
  for line in path.read_text().splitlines():
    try: row = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(row,dict) and row.get('event')=='run': rows.append(row)
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report-dir',type=Path,required=True)
  parser.add_argument('--prompts',type=int,nargs='+',default=[128,256])
  parser.add_argument('--warmups',type=int,default=4)
  parser.add_argument('--runs',type=int,default=3)
  parser.add_argument('--tokens',type=int,default=32)
  parser.add_argument('--split-down',action='store_true')
  parser.add_argument('--fused-qkv',action='store_true')
  parser.add_argument('--cpu-norm',action='store_true')
  parser.add_argument('--precision',choices=['fp16','int8'],default='fp16')
  parser.add_argument('--stock-precision',choices=['fp16','w8a8'],default='fp16')
  parser.add_argument('--stock-embed-flash',type=int,choices=[0,1],default=0)
  parser.add_argument('--cpu-pool',action='store_true')
  parser.add_argument('--cpu-fast-softmax',action='store_true')
  parser.add_argument('--cache-cleaning',action='store_true')
  parser.add_argument('--wake-helper',action='store_true')
  parser.add_argument('--quality-report',type=Path)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  matched = Path('/home/orangepi/qwen3-bench/matched')
  checkpoint = matched/'roofline/gpu/Qwen3-0.6B-matched-f16.gguf'
  binary,model = matched/'rkllm_fp16_bench',matched/'Qwen3-0.6B-fp16-3core.rkllm'
  matched_ids = args.precision==args.stock_precision=='fp16'
  quality = None
  if args.stock_precision=='w8a8': model = matched.parent/'models/qwen3-0.6b_w8a8_rk3588.rkllm'
  if not matched_ids:
    if args.quality_report is None: parser.error('Different precisions require a completed comparable-quality report')
    quality = json.loads(args.quality_report.read_text())
    if not quality.get('complete') or not quality.get('passed') or len(quality['cases'])<256:
      parser.error('Comparable-quality report must pass on at least256 scored tokens')
    if quality['checkpoint_sha256']!=digest(checkpoint) or quality['stock_model_sha256']!=digest(model):
      parser.error('Quality checkpoint/model differs')
    if any(quality['candidate_configuration'][key]!=value for key,value in
           [('precision',args.precision),('cpu_fast_softmax',args.cpu_fast_softmax)]):
      parser.error('Candidate numerical configuration differs from its quality qualification')
  args.report_dir.mkdir(parents=True,exist_ok=False)
  script = root/'extra/rockchip/benchmark_inference.py'
  sources = [p for p in (root/'tinygrad').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
  sources += [script,Path(__file__).resolve(),root/'extra/rockchip/sdk_entry.py']
  source_hashes = {str(p.relative_to(root)):digest(p) for p in sources}
  if quality is not None and any(digest(root/name)!=value for name,value in quality['candidate_source_sha256'].items()):
    parser.error('Candidate runtime source differs from its quality qualification')
  def check_source():
    if any(digest(root/name)!=value for name,value in source_hashes.items()): raise AssertionError('Measured source changed')
  report = {'complete':False,'precision':args.precision,'stock_precision':args.stock_precision,
            'checkpoint_relation':'same effective FP16 checkpoint' if matched_ids else 'declared Qwen3-0.6B; quantization differs',
            'quality_report':str(args.quality_report) if args.quality_report else None,
            'quality_sha256':digest(args.quality_report) if args.quality_report else None,
            'source_mode':'in-place hash guards',
            'split_down':args.split_down,'fused_qkv':args.fused_qkv,'cpu_norm':args.cpu_norm,
            'warmups':args.warmups,'runs':args.runs,'new_tokens':args.tokens,'cpu_affinity':[4,5,6,7],
            'source_sha256':source_hashes,'checkpoint_sha256':digest(checkpoint),
            'stock_binary_sha256':digest(binary),'stock_model_sha256':digest(model),
            'orchestrator_sha256':digest(Path(__file__)),'blocks':[]}
  target = args.report_dir/'comparison.json'
  def save(): target.write_text(json.dumps(report,indent=2)+'\n')
  save()
  env = os.environ.copy()
  env.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
             RKLLM_EMBED_FLASH=str(args.stock_embed_flash),NPU_DOMAIN_ID='1',WARMUP_RUNS=str(args.warmups))
  try:
    for length in args.prompts:
      input_path = matched/f'roofline/final-stock{length}/input.tokens'
      prompt = [int(v) for v in input_path.read_text().split()]
      if len(prompt)!=length: raise ValueError('Input token count differs from the requested workload')
      order = ['tinygrad','stock','stock','tinygrad'] if length==128 else ['stock','tinygrad','tinygrad','stock']
      for index,engine in enumerate(order):
        check_source()
        name = f'p{length}-{index}-{engine}'
        log,result = args.report_dir/(name+'.log'),args.report_dir/(name+'.json')
        if engine=='stock':
          command = ['flock','/tmp/rk3588_npu_submit.lock','taskset','-c','4-7',str(root/'.venv/bin/python'),
                     '-m','extra.rockchip.sdk_entry',str(binary),str(model),str(input_path),
                     str(args.tokens),str(args.runs)]
        else:
          command = ['taskset','-c','4-7',str(root/'.venv/bin/python'),str(script),'--checkpoint',str(checkpoint),
            '--input',str(input_path),'--tokens',str(args.tokens),'--warmups',str(args.warmups),'--runs',str(args.runs),
            '--tile-columns','24576','--fuse-projections','--cpu-attention','--native-replay','--fused-ffn','--report',str(result)]
          if args.split_down: command.append('--split-down')
          if args.fused_qkv: command.append('--fused-qkv')
          if args.cpu_norm: command.append('--cpu-norm')
          command += ['--precision',args.precision]
          if args.cpu_pool: command.append('--cpu-pool')
          if args.cpu_fast_softmax: command.append('--cpu-fast-softmax')
          if args.cache_cleaning: command.append('--cache-cleaning')
          if args.wake_helper: command.append('--wake-helper')
        block = {'name':name,'engine':engine,'prompt_tokens':length,'input_ids':prompt,'command':command,
                 'status':'running','samples':[]}
        report['blocks'].append(block)
        save()
        print(json.dumps({'event':'start','block':name}),flush=True)
        begin = time.monotonic()
        with log.open('w') as output:
          process = subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
          while process.poll() is None:
            block['samples'].append(clock_sample())
            time.sleep(.2)
          code = process.wait()
        block.update(exit_code=code,wall_seconds=time.monotonic()-begin)
        if code: raise RuntimeError(f'{name} failed; see {log}')
        if engine=='stock':
          info = [line for line in log.read_text().splitlines() if 'model_dtype:' in line]
          expected_dtype = 'FP16' if args.stock_precision=='fp16' else 'W8A8'
          if len(info)!=1 or f'model_dtype: {expected_dtype}' not in info[0]: raise AssertionError('RKLLM precision differs')
          block['sdk_model_info'] = info[0]
          rows = stock_rows(log)
          block['trials'] = rows
          measured = [r for r in rows if not r['warmup']]
        else:
          data = json.loads(result.read_text())
          if not data['complete']: raise AssertionError('Incomplete tinygrad run')
          block['result'] = data
          rows,measured = data['trials'],[r for r in data['trials'] if not r['warmup']]
        if len(rows)!=args.warmups+args.runs or len(measured)!=args.runs:
          raise AssertionError('Warmup or measurement count mismatch')
        if any(len(r['generated_ids'])!=args.tokens for r in rows): raise AssertionError('Incomplete output sequence')
        block.update(status='complete',generated_ids=rows[0]['generated_ids'],
                     median_first_token_ms=statistics.median(r['first_token_ms'] for r in measured),
                     median_decode_tps=statistics.median(r['decode_tps'] for r in measured))
        if any(r['generated_ids']!=block['generated_ids'] for r in rows): raise AssertionError('Outputs vary across trials')
        prior = [b for b in report['blocks'] if b['prompt_tokens']==length and b['status']=='complete']
        if any(b['engine']==engine and b['generated_ids']!=block['generated_ids'] for b in prior):
          raise AssertionError('Engine token IDs differ across processes')
        if matched_ids and any(b['generated_ids']!=block['generated_ids'] for b in prior): raise AssertionError('Engine token IDs differ')
        frequencies = {}
        for sample in block['samples']:
          for key,value in sample['frequencies'].items(): frequencies.setdefault(key,set()).add(value)
        block['sampled_frequencies'] = {k:sorted(v) for k,v in frequencies.items()}
        check_source()
        save()
        print(json.dumps({'event':'finished','block':name,'first_token_ms':block['median_first_token_ms'],
                          'decode_tps':block['median_decode_tps']}),flush=True)
    report['summary'] = []
    for length in args.prompts:
      groups = {engine:[b for b in report['blocks'] if b['prompt_tokens']==length and b['engine']==engine]
                for engine in ['tinygrad','stock']}
      values = {engine:{'first_token_ms':statistics.median(b['median_first_token_ms'] for b in blocks),
                        'decode_tps':statistics.median(b['median_decode_tps'] for b in blocks)} for engine,blocks in groups.items()}
      for value in values.values(): value['prefill_tps'] = length*1000/value['first_token_ms']
      report['summary'].append({'prompt_tokens':length,**values,
        'prefill_speedup':values['stock']['first_token_ms']/values['tinygrad']['first_token_ms'],
        'decode_speedup':values['tinygrad']['decode_tps']/values['stock']['decode_tps']})
    check_source()
    report['complete'] = True
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally: save()


if __name__=='__main__': main()
