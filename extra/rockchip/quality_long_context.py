"""Compare held-out next-token likelihood near the common RKLLM context maximum.

Seed the vendor KV with one normal generated token, which is not written to KV.
A 64-token probe verified the next teacher-forced logits are bitwise identical
to GET_LOGITS seeding. Hidden mode allocated excessive memory; chunked hidden
seeding changed the logits and was rejected. Quality timing is not a benchmark.
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from extra.rockchip.quality_inference import digest, sources, check_sources
from extra.rockchip.quality_runner_options import add_projection_options, projection_options, projection_option_error, fp16_reference_error


def quality_runner_hashes():
  root = Path(__file__).resolve().parents[2]
  paths = (Path(__file__).resolve(), Path(__file__).with_name('quality_inference.py').resolve(),
           Path(__file__).with_name('quality_runner_options.py').resolve())
  return {str(path.relative_to(root)):digest(path) for path in paths}


def prepare(args):
  prefix = []
  if args.checkpoint.is_dir():
    from onpu.vlm import load_tokenizer
    tokenizer = load_tokenizer(args.checkpoint)
    checkpoint = args.checkpoint/'model.safetensors'
    eos,vocab = 49279,49280
    prefix = json.loads((args.fixture/'input_ids.json').read_text())
  else:
    from tinygrad.llm.gguf import gguf_parse
    from tinygrad.llm.cli import SimpleTokenizer
    kv,_ = gguf_parse(args.checkpoint)
    tokenizer = SimpleTokenizer.from_gguf_kv(kv)
    checkpoint = args.checkpoint
    eos,vocab = kv['tokenizer.ggml.eos_token_id'],len(kv['tokenizer.ggml.tokens'])
  tokens = tokenizer.encode(args.corpus.read_text())
  rng = np.random.default_rng(20261020)
  offset = int(rng.integers(0,len(tokens)-args.context))
  sample = prefix+tokens[offset:offset+args.context-len(prefix)]
  seed = args.context-args.scored-1
  if args.scored<256 or seed<1: raise ValueError('Need at least 256 scored positions and a nonempty seed')
  cases = [{'name':'seed','start':0,'gold':-1,'input_ids':sample[:seed]}]
  cases += [{'name':f't{step:04d}','start':seed+step,'gold':sample[seed+step+1],
             'input_ids':[sample[seed+step]]} for step in range(args.scored)]
  data = {'dataset':'WikiText-2 test, PyTorch examples mirror','corpus_sha256':digest(args.corpus),
          'corpus_url':'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt',
          'sampling_seed':20261020,'corpus_token_offset':offset,'context':args.context,'scored_tokens':args.scored,
          'checkpoint':str(args.checkpoint),'checkpoint_sha256':digest(checkpoint),'vocab':vocab,
          'eos_id':eos,'max_perplexity_ratio_vs_rkllm':1.02,'max_perplexity_ratio_vs_fp16':1.02,'cases':cases}
  if prefix:
    from tinygrad import dtypes
    from tinygrad.nn.state import safe_load
    stored = safe_load(checkpoint)['model.text_model.embed_tokens.weight']
    weight = ((stored.bitcast(dtypes.uint16).numpy().astype(np.uint32)<<16).view(np.float32)
              if stored.dtype is dtypes.bfloat16 else stored.numpy().astype(np.float32))
    weight = weight.astype(np.float16).astype(np.float32)
    values = weight[np.array(sample[:seed])].copy()
    image_file = args.fixture/'vendor_image_embeddings.f32'
    image = np.fromfile(image_file,dtype=np.float32).reshape(64,576)
    slots = np.array(sample[:seed])==49190
    if slots.sum()!=64: raise AssertionError('Missing image positions')
    values[slots] = image
    seed_file = args.manifest.with_suffix('.seed-embeddings.f32')
    values.tofile(seed_file)
    data.update(fixture=str(args.fixture),seed_embeddings=str(seed_file),seed_embeddings_sha256=digest(seed_file),
                vendor_image_sha256=digest(image_file),pixels_sha256=digest(args.fixture/'pixels.npy'))
  args.manifest.write_text(json.dumps(data,indent=2)+'\n')
  args.manifest.with_suffix('.tokens').write_text(''.join(
    f"{c['name']} {c['start']} {c['gold']} "+' '.join(map(str,c['input_ids']))+'\n' for c in cases))
  print(json.dumps({k:data[k] for k in ('context','scored_tokens','corpus_token_offset','corpus_sha256')}),flush=True)


def run(args):
  data = json.loads(args.manifest.read_text())
  data.setdefault('vocab',151936)
  args.directory.mkdir(parents=True,exist_ok=False)
  report = {'complete':False,'manifest_sha256':digest(args.manifest),'harness_sha256':digest(Path(__file__)),'cases':[]}
  target = args.directory/'report.json'
  def save(): target.write_text(json.dumps(report,indent=2)+'\n')
  save()
  try:
    if args.mode=='stock':
      report.update(model=str(args.rkllm_model),model_sha256=digest(args.rkllm_model),binary_sha256=digest(args.rkllm_binary))
      command = ['flock','/tmp/rk3588_npu_submit.lock','taskset','-c','4-7',sys.executable,'-m','extra.rockchip.sdk_entry',
                 str(args.rkllm_binary),str(args.rkllm_model),str(args.manifest.with_suffix('.tokens')),
                 str(args.directory),str(data['context']),'seed-generate']
      seed_path = '-'
      if 'seed_embeddings' in data:
        seed_file = Path(data['seed_embeddings'])
        if digest(seed_file)!=data['seed_embeddings_sha256']: raise AssertionError('Seed embeddings changed')
        seed_path = str(seed_file)
      command += [str(data['vocab']),seed_path]
      report['seed_mode'] = 'full-prefix GENERATE, one output token not retained in KV'
      report['command'] = command
      save()
      with (args.directory/'runtime.log').open('w') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
    else:
      vision = args.checkpoint.is_dir()
      device = args.device or ('CPU' if vision else 'CL')
      if device=='CL' and args.precision!='fp16':
        raise ValueError('GPU qualification uses the serving FP16 path')
      if args.gpu_q8_projections and (vision or device!='CL' or args.gpu_npu or args.gpu_fuse_projections):
        raise ValueError('GPU Q8 body quality requires a GPU-only text model without projection fusion')
      if device=='CPU' and not vision and args.precision!='int8':
        raise ValueError('CPU+NPU text qualification uses INT8 projections')
      if args.cpu_packed_attention and (not vision or device!='CPU'): raise ValueError('Packed qualification requires the CPU VLM')
      os.environ['DEV'] = device
      from tinygrad import Tensor, TinyJit, Variable, Device
      from tinygrad.llm.model import Transformer
      from tinygrad.llm.kernels.mali import configure as configure_mali
      from tinygrad.llm.kernels.rockchip import configure
      input_scales = None
      def cpu_configuration(model, prefill='on'):
        return configure(model,prefill=prefill,decode='on',tile_columns=24576,prefill_rows=128,
                         fuse_projections=True,cpu_attention=True,native_replay=True,fused_ffn=True,
                         fused_qkv=True,cpu_norm=True,split_down=True,cpu_fast_softmax=True,
                         cache_cleaning=True,wake_helper=True,precision=args.precision,cpu_pool=args.precision=='int8',
                         cpu_packed_attention=args.cpu_packed_attention,cpu_fp16_qk=args.cpu_fp16_qk,cpu_int8_qk=args.cpu_int8_qk,
                         input_scales=input_scales,**projection_options(args))
      checkpoint = args.checkpoint/'model.safetensors' if vision else args.checkpoint
      if digest(checkpoint)!=data['checkpoint_sha256']: raise AssertionError('Checkpoint changed')
      if args.input_scales:
        from extra.rockchip.calibrate_text_int8 import load_scales
        input_scales = load_scales(args.input_scales,data['checkpoint_sha256'])
      hashes = sources()
      hashes.update(quality_runner_hashes())
      if vision:
        from tinygrad.llm.vision import SmolVLM
        from onpu.vlm import VisionSession
        import onpu.vlm as serving
        hashes[str(Path(serving.__file__).resolve())] = digest(Path(serving.__file__))
        vlm = SmolVLM.from_hf(args.checkpoint,data['context'])
        session = VisionSession(vlm,None)
        session.prepare_prefill()
        prefill_raw = vlm.text.greedy_embeddings_prefill_jit
        model = vlm.text
        if device=='CL':
          gpu,encoder = configure_mali(model,fuse_projections=args.gpu_fuse_projections),vlm.encode_jit
          placement = (configure(model,prefill='auto',decode='auto',tile_columns=24576,prefill_rows=gpu['prefill_chunk'])
                       if args.gpu_npu else None)
        else:
          from tinygrad.llm.kernels.rockchip_vision_pipeline import StagedVision
          gpu = None
          placement = cpu_configuration(model,'auto')
          configure(vlm.vision,prefill='on',decode='off')
          encoder = StagedVision(vlm,'ROCKCHIP',npu_connector=True,fused_mlp=True,
                                gpu_softmax=args.vision_gpu,prepared_mlp=args.vision_gpu)
      else:
        model,_ = Transformer.from_gguf(args.checkpoint,max_context=data['context'],realize=True)
        if device=='CL':
          gpu = configure_mali(model,fuse_projections=args.gpu_fuse_projections,
                               q8_head_checkpoint=args.checkpoint if args.gpu_q8_head else None,
                               q8_projections_checkpoint=args.checkpoint if args.gpu_q8_projections else None,
                               recurrent_scan=args.gpu_recurrent_scan)
          if args.gpu_q8_projections and not gpu.get('q8_projection_layers'):
            raise ValueError('Requested native Q8 body projections did not activate')
          placement = (configure(model,prefill='auto',decode='auto',tile_columns=24576,prefill_rows=gpu['prefill_chunk'])
                       if args.gpu_npu else None)
        else:gpu,placement = None,cpu_configuration(model)
      checkpoint_sha256 = digest(checkpoint)
      scales_sha256 = digest(args.input_scales) if args.input_scales else None
      scale_metadata = json.loads(args.input_scales.read_text()) if args.input_scales else None
      report.update(source_sha256=hashes,checkpoint_path=str(args.checkpoint),model_checkpoint_path=str(checkpoint),
                    checkpoint_sha256=checkpoint_sha256,gpu_kernels=gpu,placement=placement,
                    precision=args.precision,device=device,vision_gpu=args.vision_gpu,gpu_npu=args.gpu_npu,
                    gpu_fuse_projections=args.gpu_fuse_projections,gpu_q8_head=args.gpu_q8_head,
                    gpu_q8_projections=args.gpu_q8_projections,gpu_recurrent_scan=args.gpu_recurrent_scan,
                    requested_configuration={'mode':args.mode,'precision':args.precision,'device':device,
                      'context':data['context'],'share_projection_weights':args.share_projection_weights,
                      'cpu_prefill_pool':args.cpu_prefill_pool,'cpu_recurrent_scan':args.cpu_recurrent_scan,
                      'gpu_recurrent_scan':args.gpu_recurrent_scan,'gpu_npu':args.gpu_npu,'cpu_norm':device=='CPU',
                      'cpu_packed_attention':args.cpu_packed_attention,
                      'cpu_fp16_qk':args.cpu_fp16_qk,'cpu_int8_qk':args.cpu_int8_qk,
                      'input_scales_path':str(args.input_scales) if args.input_scales else None,
                      'input_scales_sha256':scales_sha256},
                    input_scales_sha256=scales_sha256,
                    calibration_provenance=None if scale_metadata is None else {
                      'path':str(args.input_scales),'sha256':scales_sha256,
                      'schema_version':scale_metadata.get('schema_version'),
                      'checkpoint_sha256':scale_metadata.get('checkpoint_sha256'),
                      'dataset_sha256':scale_metadata.get('dataset_sha256'),
                      'calibration_prompts':scale_metadata.get('calibration_prompts'),
                      'alpha':scale_metadata.get('alpha'),'skip_down':scale_metadata.get('skip_down'),
                      'capture_mode':scale_metadata.get('capture_mode'),
                      'source_sha256':scale_metadata.get('source_sha256'),
                      'harness_sha256':scale_metadata.get('harness_sha256'),
                      'group_count':len(scale_metadata.get('groups',{}))})
      save()
      jit = TinyJit(lambda tokens,start:model.logits(tokens,start).realize())
      for case in data['cases']:
        begin = time.perf_counter()
        ids,start = case['input_ids'],case['start']
        if case['name']=='seed':
          if vision:
            fixture = Path(data['fixture'])
            if digest(fixture/'pixels.npy')!=data['pixels_sha256']: raise AssertionError('Image changed')
            pixels = Tensor(np.load(fixture/'pixels.npy'),device=device).realize()
            if args.precision=='int8':
              Device['ROCKCHIP'].reset()
            vision_start = time.perf_counter()
            image = encoder(pixels).realize()
            Device[device].synchronize()
            report['vision_seconds'] = time.perf_counter()-vision_start
            print(json.dumps({'event':'vision','seconds':report['vision_seconds']}),flush=True)
            embeddings = vlm.embed_prompt(ids,image).realize()
            session.prefill(embeddings,greedy=True,tokens=ids).item()
            if args.cpu_packed_attention:
              captured = prefill_raw.captured
              report['prefill_native'] = {'runs':getattr(captured,'native_runs',0),'reason':getattr(captured,'reason',None)}
              if captured is None or captured.native is None or report['prefill_native']['runs']<1:
                raise AssertionError('Packed full-model prefill did not execute native replay')
              headers = [block.prefill_attention.packed[:4].bitcast('int32').numpy().tolist() for block in model.blk]
              if len(headers)!=30 or any(h[0]!=len(ids) or h[1]>128 for h in headers):
                raise AssertionError(f'Packed cache did not preserve incremental state: {headers}')
              report['packed_headers_after_seed'] = headers
          else:
            # Serving binds the last partial chunk symbolically. A sequence of
            # fixed-width tensors would violate the JIT signature at that tail.
            generation = model.generate(ids.copy(),chunk_size=gpu['prefill_chunk'] if gpu else 128,temperature=0)
            try: next(generation)
            finally: generation.close()
            if model._cached_tokens!=ids: raise AssertionError('Seed KV history differs')
        else:
          x = Tensor([ids],device=device,dtype='int32').realize()
          position = Variable('start_pos',0,data['context']-1).bind(start)
          result = jit(x,position).numpy().ravel()
          if result.size!=data['vocab'] or not np.isfinite(result).all(): raise AssertionError('Invalid vocabulary')
          result.tofile(args.directory/(case['name']+'.f32'))
        check_sources(hashes)
        print(json.dumps({'event':'quality','name':case['name'],'start':start,'seconds':time.perf_counter()-begin}),flush=True)
    for case in data['cases'][1:]:
      path = args.directory/(case['name']+'.f32')
      if path.stat().st_size!=data['vocab']*4: raise AssertionError('Incomplete vocabulary dump')
      report['cases'].append({'name':case['name'],'sha256':digest(path)})
    report['complete'] = True
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally: save()
  return model if args.mode=='candidate' else None


def compare(args):
  data = json.loads(args.manifest.read_text())
  data.setdefault('vocab',151936)
  metadata = [json.loads((p/'report.json').read_text()) for p in (args.stock,args.candidate)]
  if any(not r['complete'] or r['manifest_sha256']!=digest(args.manifest) for r in metadata):
    raise AssertionError('Incomplete or different quality requests')
  reference=getattr(args,'reference_engine','rkllm')
  if reference not in ('rkllm','fp16'):raise ValueError('Unknown quality reference engine')
  if reference=='fp16':
    if error:=fp16_reference_error(metadata[0],metadata[1],data):raise AssertionError(error)
    if data.get('scored_tokens',0)<256 or len(data['cases'])-1!=data['scored_tokens']:
      raise AssertionError('Local FP16 quality requires at least 256 declared held-out positions')
  elif (metadata[0].get('device')=='CL' or
        any(not isinstance(metadata[0].get(key),str) or len(metadata[0][key])!=64 for key in ('model_sha256','binary_sha256'))):
    raise AssertionError('RKLLM reference requires the SDK model and binary hashes')
  limit=data.get('max_perplexity_ratio_vs_'+reference,1.02)
  if not isinstance(limit,(int,float)) or not math.isfinite(limit) or not 0<limit<=1.02:
    raise AssertionError('Comparable-quality limit must not exceed 1.02')
  mask = np.arange(data['vocab'])!=data['eos_id']
  rows = []
  for case in data['cases'][1:]:
    row = {'name':case['name'],'start':case['start'],'gold':case['gold']}
    if not mask[case['gold']]: raise AssertionError('Gold token excluded by SDK EOS policy')
    for engine,path,meta in zip((reference,'onpu'),(args.stock,args.candidate),metadata):
      file = path/(case['name']+'.f32')
      if digest(file)!=next(r['sha256'] for r in meta['cases'] if r['name']==case['name']): raise AssertionError('Dump changed')
      values = np.fromfile(file,dtype=np.float32).astype(np.float64)
      if values.size!=data['vocab'] or not np.isfinite(values[mask]).all(): raise AssertionError('Invalid logits')
      maximum = values[mask].max()
      row[engine] = {'nll':float(maximum+np.log(np.exp(values[mask]-maximum).sum())-values[case['gold']]),
                     'argmax':int(np.where(mask,values,-np.inf).argmax())}
    rows.append(row)
  means = {e:float(np.mean([r[e]['nll'] for r in rows])) for e in (reference,'onpu')}
  ratio = math.exp(means['onpu']-means[reference])
  report = {'complete':True,'manifest_sha256':digest(args.manifest),'positions':len(rows),'mean_nll':means,
            'conditional_perplexity':{e:math.exp(v) for e,v in means.items()},'perplexity_ratio_vs_'+reference:ratio,
            'reference_engine':reference,'max_perplexity_ratio':limit,'comparison_harness_sha256':digest(Path(__file__)),
            'passed':ratio<=limit,'cases':rows,'provenance':metadata}
  args.directory.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k not in ('cases','provenance')}),flush=True)
  if not report['passed']: raise AssertionError('Comparable-quality gate failed')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('mode',choices=['prepare','stock','candidate','compare'])
  parser.add_argument('--manifest',type=Path,required=True)
  for name in ('checkpoint','corpus','directory','rkllm-model','rkllm-binary','stock','candidate','fixture'):
    parser.add_argument('--'+name,type=Path)
  parser.add_argument('--context',type=int,default=16384)
  parser.add_argument('--scored',type=int,default=256)
  parser.add_argument('--precision',choices=['fp16','int8'],default='fp16')
  parser.add_argument('--device',choices=['CPU','CL'])
  parser.add_argument('--gpu-npu',action='store_true',help='Measure optional NPU placement for a GPU profile')
  parser.add_argument('--gpu-fuse-projections',action='store_true')
  parser.add_argument('--gpu-q8-head',action='store_true')
  parser.add_argument('--gpu-q8-projections',action='store_true',help='Qualify native Q8_0 body projections on Mali')
  parser.add_argument('--gpu-recurrent-scan',action='store_true',help='Use the qualified Mali recurrent scan for an FP16 reference')
  parser.add_argument('--reference-engine',choices=['rkllm','fp16'],default='rkllm',
                      help='Engine supplying the --stock logits directory; FP16 is explicitly a local GPU comparison')
  parser.add_argument('--input-scales',type=Path)
  parser.add_argument('--cpu-packed-attention',action='store_true')
  parser.add_argument('--vision-gpu',action='store_true')
  parser.add_argument('--cpu-fp16-qk',action='store_true')
  parser.add_argument('--cpu-int8-qk',action='store_true')
  add_projection_options(parser)
  args = parser.parse_args()
  if args.gpu_fuse_projections and args.gpu_npu:parser.error('Joint GPU qualification currently requires assistance Off')
  if (args.gpu_fuse_projections or args.gpu_q8_head or args.gpu_npu) and args.device=='CPU':parser.error('GPU options require CL')
  if args.input_scales and (args.device!='CPU' or args.precision!='int8'):parser.error('Input scaling requires CPU/NPU INT8')
  if args.gpu_q8_head and (args.checkpoint is None or args.checkpoint.is_dir()):parser.error('Q8 head qualification requires a GGUF checkpoint')
  if args.gpu_recurrent_scan and (args.mode!='candidate' or args.device=='CPU' or args.precision!='fp16'
                                 or args.checkpoint is None or args.checkpoint.is_dir()):
    parser.error('GPU recurrent scan requires a text GGUF FP16 GPU candidate/reference run')
  projection_error = projection_option_error(mode=args.mode,precision=args.precision,device=args.device,
    checkpoint_is_text=args.checkpoint is not None and not args.checkpoint.is_dir(),
    share_projection_weights=args.share_projection_weights,cpu_prefill_pool=args.cpu_prefill_pool,
    cpu_recurrent_scan=args.cpu_recurrent_scan)
  if projection_error:parser.error(projection_error)
  if args.mode=='prepare': prepare(args)
  elif args.mode=='compare': compare(args)
  else: run(args)


if __name__=='__main__': main()
