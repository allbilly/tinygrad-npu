"""Image-conditioned quality gate for the shared SmolVLM checkpoint.

The held-out cases exclude the demo calibration/performance image. Score only
positions up to the first reference EOS. Compare pooled conditional perplexity
and the same multiple-choice questions; never count forced argmaxes as answers.
"""
import argparse, hashlib, json, math, os, re
from pathlib import Path
import numpy as np

def digest(path):
  with Path(path).open('rb') as file: return hashlib.file_digest(file,'sha256').hexdigest()

def source_hashes():
  source = Path(__file__).resolve().parents[2]
  paths = [*sorted(p for p in (source/'tinygrad').rglob('*') if p.suffix in ('.py','.c')),Path(__file__)]
  return {str(path):digest(path) for path in paths}

def tokenizer(root):
  from tinygrad.llm.cli import SimpleTokenizer
  data = json.loads((root/'hf/tokenizer.json').read_text())
  special = {item['content']:item['id'] for item in data['added_tokens']}
  normal = {token:index for token,index in data['model']['vocab'].items() if token not in special}
  return SimpleTokenizer(normal,special,preset='smollm',eos_id=49279)

def manifest(root):
  lines = []
  for case in json.loads((root/'quality-cases.json').read_text()):
    directory = root/'quality'/case['id']
    report = json.loads((directory/'reference.json').read_text())
    count = report['quality_positions_before_or_at_eos']
    gold = report['generated_ids'][:count]
    if not 1<=count<=16 or gold[-1]!=49279 and count!=16: raise ValueError('Invalid pre-EOS reference positions')
    lines.append(' '.join([case['id'],str(directory),str(count),*map(str,gold)]))
  (root/'quality-manifest.txt').write_text('\n'.join(lines)+'\n')

def candidate(args):
  os.environ['DEV'] = args.text_device
  os.environ['ROCKCHIP_BINDING_CACHE'] = '4096'
  from tinygrad import Device, Tensor, TinyJit, UOp
  from tinygrad.llm.vision import SmolVLM
  from tinygrad.llm.kernels.rockchip import configure
  from tinygrad.llm.kernels.rockchip_vision_pipeline import StagedVision, load_input_scales
  root, output = args.root,args.output
  output.mkdir(parents=True,exist_ok=True)
  report = {'complete':False,'checkpoint_sha256':digest(root/'hf/model.safetensors'),
            'precision':args.precision,'context':args.context,'text_device':args.text_device,'gpu_npu':args.gpu_npu,
            'vision':args.vision,'vision_precision':args.vision_precision,
            'shared_gpu_softmax':args.shared_gpu_softmax,
            'prepared_vision_mlp':args.prepared_vision_mlp,
            'vision_calibration':str(args.vision_calibration) if args.vision_calibration else None,
            'calibration_sha256':digest(args.vision_calibration) if args.vision_calibration else None,
            'source_sha256':source_hashes(),'input_sha256':{},'reset_before_vision':args.reset_before_vision,'cases':{}}
  def save(): (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  save()
  model = SmolVLM.from_hf(root/'hf',args.context,vision_device='CL' if args.vision=='cl' else 'CPU')
  session, scalar_jit, scalar_decoder = None,None,None
  if args.context>512:
    from onpu.vlm import VisionSession
    import onpu.vlm as serving
    from tinygrad.llm.kernels.rockchip_decode import prepare_scalar_decode
    report['source_sha256'][str(Path(serving.__file__).resolve())] = digest(serving.__file__)
    session = VisionSession(model,None)
    session.prepare_prefill()
    if args.text_device=='CPU': scalar_jit = prepare_scalar_decode(model.text)
  if args.text_device=='CL':
    from tinygrad.llm.kernels.mali import configure as configure_mali
    report['gpu_kernels'] = configure_mali(model.text)
  if args.vision=='staged':
    scales = load_input_scales(args.vision_calibration,report['checkpoint_sha256']) if args.vision_calibration else None
    configure(model.vision,prefill='on',decode='off',precision=args.vision_precision,input_scales=scales,batch_prefill=args.batch_vision_prefill)
    encode = StagedVision(model,args.staged_attention,npu_connector=args.npu_connector,fused_mlp=args.fused_vision_mlp,
                          gpu_softmax=args.shared_gpu_softmax,prepared_mlp=args.prepared_vision_mlp)
  else: encode = model.encode_jit
  # The quality path returns the full vocabulary; generation uses separate JITs.
  model.text.rollout_jit = TinyJit(lambda tokens,position:model.text.logits(tokens,position).realize())
  if args.text_device=='CL':
    report['text_placement'] = (configure(model.text,prefill='auto',decode='auto',tile_columns=24576,prefill_rows=128)
                                if args.gpu_npu else None)
  else:
    report['text_placement'] = configure(model.text,prefill='auto' if session else 'on',decode='on',precision=args.precision,tile_columns=24576,
      fuse_projections=True,cpu_attention=True,native_replay=True,fused_ffn=True,cpu_norm=True,
      cpu_pool=args.precision=='int8' or session is None,cpu_fast_softmax=True,cache_cleaning=True,wake_helper=True,
      cpu_packed_attention=args.cpu_packed_attention,cpu_fp16_qk=args.cpu_fp16_qk,cpu_int8_qk=args.cpu_int8_qk,
      **({'split_down':True,'fused_qkv':True,'prefill_rows':128} if session else {}))
  if scalar_jit is not None:
    from tinygrad.llm.kernels.rockchip_decode import ScalarDecoder
    from tinygrad.llm.kernels.rockchip_idle import idle_helper
    from tinygrad.llm.kernels.rockchip_pool import attention_pool
    scalar_decoder = ScalarDecoder(model.text,scalar_jit,pool=attention_pool() if args.precision=='int8' else None,helper=idle_helper())
  prefill = session.prefill if session else model.text.embeddings_prefill_jit
  greedy_prefill = (lambda values:session.prefill(values,greedy=True)) if session else model.text.greedy_embeddings_prefill_jit
  def decode(token,pos):
    position = UOp.variable('position',1,args.context-1).bind(pos)
    return model.text.rollout_jit(Tensor([[token]]).contiguous(),position).numpy()[0]
  def gpu_greedy(token,pos):
    position = UOp.variable('start_pos',0,args.context-1).bind(pos)
    return int(model.text.greedy_rollout_jit(Tensor([[token]],device='CL').contiguous(),position).item())
  for case in json.loads((root/'quality-cases.json').read_text()):
    directory = root/'quality'/case['id']
    for name in ('pixels.npy','input_ids.json','reference.json','teacher_logits.npy','image_embeddings.npy'):
      report['input_sha256'][str(directory/name)] = digest(directory/name)
    reference = json.loads((directory/'reference.json').read_text())
    ids, gold = reference['input_ids'],reference['generated_ids'][:reference['quality_positions_before_or_at_eos']]
    if args.reset_before_vision: Device['ROCKCHIP'].reset()
    image = encode(Tensor(np.load(directory/'pixels.npy'),device='CL' if args.vision=='cl' else 'CPU').realize()).realize()
    image.numpy().tofile(output/(case['id']+'.image.f32'))
    embeddings = model.embed_prompt(ids,image).realize()
    first = prefill(embeddings).numpy()[0]
    logits = [first]
    for step in range(1,len(gold)): logits.append(decode(gold[step-1],len(ids)+step-1))
    np.stack(logits).astype(np.float32).tofile(output/(case['id']+'.logits.f32'))
    value = prefill(embeddings).numpy()[0]
    first_token = int(greedy_prefill(embeddings).item())
    if first_token!=int(value.argmax()):raise RuntimeError('Greedy embedding prefill differs from full logits')
    generated = []
    for step in range(16):
      if step==0: token = first_token
      elif scalar_decoder: token = scalar_decoder(generated[-1],len(ids)+step-1,1)[0]
      elif args.text_device=='CL': token = gpu_greedy(generated[-1],len(ids)+step-1)
      else: token = int(value.argmax())
      generated.append(token)
      if token==49279 or step==15: break
      if scalar_decoder is None and args.text_device=='CPU': value = decode(token,len(ids)+step)
    report['cases'][case['id']] = {'positions':len(gold),'generated_ids':generated}
    save()
    print(json.dumps({'event':'case','name':case['id'],'positions':len(gold),'generated_ids':generated}),flush=True)
  if any(digest(path)!=value for path,value in {**report['source_sha256'],**report['input_sha256']}.items()):
    raise RuntimeError('Quality sources or inputs changed during evaluation')
  report['complete'] = True
  report['greedy_prefill_checked'] = True
  report['scalar_generated_answers'] = scalar_decoder is not None
  report['gpu_generated_answers'] = args.text_device=='CL'
  save()

def log_probs(values):
  values = values.astype(np.float64)
  if not np.isfinite(values).all(): raise ValueError('Nonfinite quality logits')
  shifted = values-values.max(axis=-1,keepdims=True)
  return shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))

def answer(text):
  text = text.replace('<end_of_utterance>','').strip()
  first = re.match(r'^([A-D])(?:[.\s):]|$)',text,re.IGNORECASE)
  if first: return first[1].upper()
  found = re.search(r'(?:answer(?:\s+is)?|option)\s*[:=]?\s*([A-D])(?:[.\s):]|$)',text,re.IGNORECASE)
  return found[1].upper() if found else None

def compare(args):
  root, vendor, current = args.root,args.vendor,args.output
  report = json.loads((current/'report.json').read_text())
  vendor_report = json.loads((vendor/'report.json').read_text())
  if not report['complete'] or report['checkpoint_sha256']!=digest(root/'hf/model.safetensors'):
    raise ValueError('Incomplete or mismatched candidate')
  if any(digest(path)!=value for path,value in {**report['source_sha256'],**report['input_sha256']}.items()):
    raise ValueError('Candidate quality sources or inputs changed')
  if not vendor_report['complete'] or vendor_report['checkpoint_sha256']!=report['checkpoint_sha256']:
    raise ValueError('Incomplete or mismatched vendor quality run')
  if any(digest(path)!=value for path,value in {**vendor_report['source_sha256'],**vendor_report['output_sha256']}.items()):
    raise ValueError('Vendor quality sources or outputs changed')
  tok = tokenizer(root)
  vendor_cases = {}
  for line in (vendor/'runtime.log').read_text(errors='replace').splitlines():
    if line.startswith('{'):
      row = json.loads(line)
      if row.get('event')=='case': vendor_cases[row['name']] = row
  losses = {name:[] for name in ('reference','rkllm','onpu')}
  correct = {'reference':0,'rkllm':0,'onpu':0}
  kl = {'rkllm':[],'onpu':[]}
  image_error = {'rkllm':[],'onpu':[]}
  cases = []
  for case in json.loads((root/'quality-cases.json').read_text()):
    key = case['id']
    ref = json.loads((root/'quality'/key/'reference.json').read_text())
    count = ref['quality_positions_before_or_at_eos']
    gold = ref['generated_ids'][:count]
    arrays = {'reference':np.load(root/'quality'/key/'teacher_logits.npy')[:count],
              'rkllm':np.fromfile(vendor/(key+'.logits.f32'),dtype=np.float32).reshape(count,49280),
              'onpu':np.fromfile(current/(key+'.logits.f32'),dtype=np.float32).reshape(count,49280)}
    probabilities = {name:log_probs(array) for name,array in arrays.items()}
    row = {'id':key,'target':case['target'],'positions':count,'answers':{}}
    image_reference = np.load(root/'quality'/key/'image_embeddings.npy').astype(np.float64)
    row['image_nrmse'] = {}
    for name,directory in [('rkllm',vendor),('onpu',current)]:
      image = np.fromfile(directory/(key+'.image.f32'),dtype=np.float32).reshape(image_reference.shape).astype(np.float64)
      if not np.isfinite(image).all(): raise ValueError('Nonfinite image features')
      error = float(np.linalg.norm(image-image_reference)/max(np.linalg.norm(image_reference),1e-30))
      image_error[name].append(error)
      row['image_nrmse'][name] = error
    generated = {'reference':ref['generated_ids'],'rkllm':vendor_cases[key]['generated_ids'],
                 'onpu':report['cases'][key]['generated_ids']}
    for name,values in probabilities.items():
      losses[name].extend((-values[np.arange(count),gold]).tolist())
      text = tok.decode(generated[name])
      selected = answer(text)
      correct[name] += selected==case['target']
      row['answers'][name] = {'text':text,'selected':selected,'correct':selected==case['target']}
      if name!='reference': kl[name].extend((np.exp(probabilities['reference'])*(probabilities['reference']-values)).sum(axis=-1).tolist())
    cases.append(row)
  perplexity = {name:math.exp(float(np.mean(values))) for name,values in losses.items()}
  ratio = perplexity['onpu']/perplexity['rkllm']
  image_passed = all(a<=b*1.1+1e-5 for a,b in zip(image_error['onpu'],image_error['rkllm']))
  result = {'complete':True,'positions':len(losses['reference']),'case_count':len(cases),'conditional_perplexity':perplexity,
            'onpu_to_rkllm_ppl_ratio':ratio,'max_ppl_ratio':1.02,'correct':correct,'mean_kl_to_fp32':{k:float(np.mean(v)) for k,v in kl.items()},
            'image_nrmse':{k:{'mean':float(np.mean(v)),'max':max(v)} for k,v in image_error.items()},
            'max_image_error_ratio':1.1,'image_gate_passed':image_passed,
            'passed':ratio<=1.02 and correct['onpu']>=correct['rkllm'] and image_passed,'cases':cases,
            'scope':'20 vendor-provided MMBench cases, greedy FP32 teacher prefixes, before first EOS; limited quality proxy'}
  (current/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
  if not result['passed']: raise AssertionError('Comparable VLM quality gate failed')

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('mode',choices=['manifest','candidate','compare'])
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--output',type=Path)
  parser.add_argument('--vendor',type=Path)
  parser.add_argument('--vision',choices=['cl','staged'],default='staged')
  parser.add_argument('--precision',choices=['fp16','int8'],default='int8')
  parser.add_argument('--context',type=int,default=512)
  parser.add_argument('--text-device',choices=['CPU','CL'],default='CPU')
  parser.add_argument('--gpu-npu',action='store_true')
  parser.add_argument('--cpu-packed-attention',action='store_true')
  parser.add_argument('--cpu-fp16-qk',action='store_true')
  parser.add_argument('--cpu-int8-qk',action='store_true')
  parser.add_argument('--reset-before-vision',action='store_true')
  parser.add_argument('--vision-precision',choices=['fp16','int8'],default='fp16')
  parser.add_argument('--vision-calibration',type=Path)
  parser.add_argument('--staged-attention',choices=['CL','ROCKCHIP'],default='CL')
  parser.add_argument('--npu-connector',action='store_true')
  parser.add_argument('--fused-vision-mlp',action='store_true')
  parser.add_argument('--shared-gpu-softmax',action='store_true')
  parser.add_argument('--prepared-vision-mlp',action='store_true')
  parser.add_argument('--batch-vision-prefill',action='store_true')
  args = parser.parse_args()
  if not 64<=args.context<=8192: parser.error('SmolVLM context must be between 64 and 8192')
  if args.text_device=='CL' and args.precision!='fp16': parser.error('GPU text qualification uses FP16')
  if args.mode=='manifest': manifest(args.root)
  elif args.mode=='candidate': candidate(args)
  else: compare(args)

if __name__ == '__main__': main()
