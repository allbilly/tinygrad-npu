"""Benchmark raw-token requests against a matched RKLLM log, with explicit NPU placement.

Use matched checkpoint/input IDs, context, output count and quality for both engines.
Record precision, device placement, affinity and clocks explicitly.
Loading, packing, full warmups and KV reset are excluded. EOS does not stop generation.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, resource, statistics, time
from collections import Counter
from pathlib import Path


def observe(event:str, index:int): pass  # Optional external diagnostic observer; no production execution changes.


def clocks():
  paths = [*Path('/sys/class/devfreq').glob('*/cur_freq'),
           *Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')]
  return {str(path):path.read_text().strip() for path in paths}


def stock_ids(path, count):
  rows = []
  for line in path.read_text().splitlines():
    try: row = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(row,dict) and row.get('event')=='run': rows.append(row['generated_ids'][:count])
  if not rows or len(rows[0])!=count or any(row!=rows[0] for row in rows):
    raise ValueError('Stock reference needs consistent complete token sequences')
  return rows[0]


def source_snapshot(root:Path) -> dict[str,str]:
  """Hash numerical sources without importing a model or opening a device.

  Match paired qualification's package-wide coverage so new kernel modules,
  including checkpoint decoders, cannot silently escape report provenance.
  """
  files = sorted(p for p in (root/'tinygrad').rglob('*') if p.is_file() and p.suffix in ('.py','.c','.h'))
  files.append(root/'extra/rockchip/benchmark_inference.py')
  return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def main(model_override=None,gpu_kernels_override=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--defer-weight-realization',action='store_true',help='Materialize CPU weights only when their consumers run')
  parser.add_argument('--file-backed-cpu-weights',action='store_true',help='Borrow packed file mappings for deferred CPU weight decoding')
  parser.add_argument('--input',type=Path,required=True)
  parser.add_argument('--device',choices=['CPU','CL'],default='CPU')
  parser.add_argument('--prefill',choices=['off','auto','on'],default='on')
  parser.add_argument('--decode',choices=['off','auto','on'],default='on')
  parser.add_argument('--tile-columns',type=int,default=4096)
  parser.add_argument('--fuse-projections',action='store_true')
  parser.add_argument('--share-projection-weights',action='store_true',help='Opt into shared packed INT8 combined projections')
  parser.add_argument('--cpu-attention',action='store_true')
  parser.add_argument('--npu-prefill-attention',action='store_true',help='Experimental prepared FP16 causal NPU attention with CPU fallback')
  parser.add_argument('--cpu-recurrent-scan',action='store_true',help='Experimentally batch GatedDeltaNet/RWKV7 prefill through FP32 CPU scans')
  parser.add_argument('--native-replay',action='store_true')
  parser.add_argument('--fused-ffn',action='store_true')
  parser.add_argument('--fused-gelu-ffn',action='store_true',help='Fuse bounded raw INT8 GELU FFNs during prefill')
  parser.add_argument('--fused-erf-gelu-ffn',action='store_true',help='Fuse raw INT8 erf GELU and per-K input packing')
  parser.add_argument('--fused-erf-gelu-decode',action='store_true',help='Opt into raw erf GELU packing for one-token decode')
  parser.add_argument('--fused-swiglu-ffn',action='store_true',help='Fuse raw INT8 SwiGLU and split-K input packing during prefill')
  parser.add_argument('--fused-swiglu-decode',action='store_true',help='Opt into raw INT8 SwiGLU packing for one-token decode')
  parser.add_argument('--split-down',action='store_true')
  parser.add_argument('--fused-qkv',action='store_true')
  parser.add_argument('--fused-qkv-without-norm',action='store_true',help='Extend fused QKV preparation to models without head normalization')
  parser.add_argument('--cpu-norm',action='store_true')
  parser.add_argument('--cpu-static-gemm',action='store_true',help='Prepare static CPU projection weights before JIT tracing')
  parser.add_argument('--cpu-mla-weights',action='store_true',help='Prepare contiguous FP16 CPU MLA tables before JIT tracing')
  parser.add_argument('--cpu-mla-attention',action='store_true',help='Vectorize CPU MLA attention dots while retaining generic softmax')
  parser.add_argument('--cpu-mla-projections',action='store_true',help='Use prepared FP16 tables for batched FP32 CPU MLA projections')
  parser.add_argument('--cpu-experts',action='store_true',help='Group routed CPU expert dots with packed FP16 weights')
  parser.add_argument('--cpu-expert-prefill-rows',type=int,choices=(4,6),default=4,help='Rows per multi-token CPU expert tile')
  parser.add_argument('--npu-experts',action='store_true',help='Experimental compact INT8 MoE experts for CPU/NPU inference')
  parser.add_argument('--npu-expert-activation-residual',action='store_true',help='Use two INT8 products for expert activations')
  parser.add_argument('--npu-expert-weight-residual',action='store_true',help='Add a third expert product for quantized weight residuals')
  parser.add_argument('--npu-expert-cpu-layers',type=lambda s:tuple(int(i) for i in s.split(',') if i),default=(),
                      help='Keep these zero-based transformer block expert projections on CPU')
  parser.add_argument('--npu-expert-cpu-down-layers',type=lambda s:tuple(int(i) for i in s.split(',') if i),default=(),
                      help='Keep expert down projections on CPU for these transformer blocks')
  parser.add_argument('--npu-mla-prefill',action='store_true',help='Use grouped NPU MLA projections during prefill')
  parser.add_argument('--npu-mla-decode',action='store_true',help='Use grouped NPU MLA projections during decode')
  parser.add_argument('--npu-mla-precision',choices=['fp16','int8','int8_residual'],default='fp16',
                      help='Experimental grouped MLA precision; int8_residual uses three scaled INT8 products')
  parser.add_argument('--npu-mla-projections',choices=['both','query','value'],default='both',
                      help='Select MLA projections for NPU placement; other projections keep their CPU dot')
  parser.add_argument('--cpu-gelu',action='store_true',help='Vectorize CPU GELU-tanh gated activations')
  parser.add_argument('--cpu-erf-gelu',action='store_true',help='Opt into vector erf-GELU gating for CPU prefill')
  parser.add_argument('--cpu-softcap',action='store_true',help='Vectorize the final CPU logit soft cap')
  parser.add_argument('--precision',choices=['fp16','int8'],default='fp16')
  parser.add_argument('--cpu-pool',action='store_true')
  parser.add_argument('--cpu-prefill-pool',action='store_true',help='Experimentally reuse parked workers during CPU prefill attention')
  parser.add_argument('--cpu-fast-softmax',action='store_true')
  parser.add_argument('--cache-cleaning',action='store_true')
  parser.add_argument('--wake-helper',action='store_true')
  parser.add_argument('--context',type=int,default=512)
  parser.add_argument('--tokens',type=int,default=32)
  parser.add_argument('--warmups',type=int,default=2)
  parser.add_argument('--warmup-tokens',type=int,help='Warm the bounded graph with a shorter prompt before a long diagnostic')
  parser.add_argument('--chunk-size',type=int,help='Bounded prefill rows; defaults to the prepared Mali chunk on that GPU')
  parser.add_argument('--runs',type=int,default=3)
  parser.add_argument('--gpu-fuse-projections',action='store_true')
  parser.add_argument('--gpu-recurrent-scan',action='store_true')
  parser.add_argument('--gpu-q8-head',action='store_true')
  parser.add_argument('--gpu-q8-projections',action='store_true',help='Use native Q8_0 body projections on Mali; experimental')
  parser.add_argument('--gpu-no-weight-packing',action='store_true',help='Use FP16 GPU weights directly without a second GEMM copy')
  parser.add_argument('--input-scales',type=Path)
  parser.add_argument('--stock-reference',type=Path)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  if args.npu_prefill_attention and (args.device!='CPU' or not args.cpu_attention or args.prefill=='off'):
    raise ValueError('NPU prefill attention requires CPU attention hooks and active NPU prefill')
  if args.cpu_erf_gelu and args.device!='CPU':raise ValueError('CPU erf GELU requires device CPU')
  if args.fused_erf_gelu_ffn and (args.device!='CPU' or args.precision!='int8' or args.prefill==args.decode=='off'):
    raise ValueError('Fused erf GELU requires CPU INT8 and an active NPU phase')
  if args.fused_erf_gelu_decode and (not args.fused_erf_gelu_ffn or args.decode=='off'):
    raise ValueError('Fused erf GELU decode requires fused_erf_gelu_ffn and an active NPU decode phase')
  if args.fused_swiglu_decode and (args.device!='CPU' or args.precision!='int8' or not args.fused_swiglu_ffn or args.decode=='off'):
    parser.error('Fused SwiGLU decode requires CPU INT8, --fused-swiglu-ffn and an active NPU decode phase')
  if args.cpu_static_gemm and args.device!='CPU':parser.error('Static CPU projections require CPU inference')
  if args.cpu_mla_weights and args.device!='CPU':parser.error('CPU MLA weight preparation requires CPU inference')
  if args.cpu_mla_attention and args.device!='CPU':parser.error('CPU MLA attention requires CPU inference')
  if args.cpu_mla_projections and args.device!='CPU':parser.error('CPU MLA projections require CPU inference')
  if args.cpu_mla_projections and (args.npu_mla_prefill or args.npu_mla_decode):parser.error('Choose CPU or NPU MLA projections')
  if (args.npu_mla_prefill or args.npu_mla_decode) and args.device!='CPU':parser.error('Grouped MLA projections require CPU/NPU inference')
  if args.npu_experts and args.device!='CPU':parser.error('NPU experts require CPU/NPU inference')
  if args.cpu_gelu and args.device!='CPU':parser.error('CPU GELU requires CPU inference')
  if args.cpu_softcap and args.device!='CPU':parser.error('CPU soft cap requires CPU inference')
  if args.fused_gelu_ffn and (args.device!='CPU' or args.precision!='int8'):parser.error('Fused GELU FFN requires CPU INT8 inference')
  if args.file_backed_cpu_weights and (args.device!='CPU' or not args.defer_weight_realization):
    parser.error('File-backed CPU weights require CPU inference and deferred weight realization')
  if args.share_projection_weights and (args.device!='CPU' or args.precision!='int8' or not args.fuse_projections
                                       or args.prefill!='on' or args.decode!='on'):
    parser.error('Shared projection weights require CPU INT8 with fused projections and both NPU phases on')
  if args.gpu_fuse_projections and (args.device!='CL' or args.prefill!='off' or args.decode!='off'):
    parser.error('GPU projection fusion currently requires a GPU-only benchmark')
  if args.gpu_q8_head and args.device!='CL':parser.error('GPU Q8 head requires CL')
  if args.gpu_q8_projections and (args.device!='CL' or args.prefill!='off' or args.decode!='off' or args.gpu_fuse_projections):
    parser.error('GPU Q8 body requires GPU-only inference without projection fusion')
  if args.gpu_recurrent_scan and args.device!='CL':parser.error('GPU recurrent scan requires CL')
  if args.cpu_recurrent_scan and args.device!='CPU':parser.error('CPU recurrent scan requires CPU')
  if args.gpu_no_weight_packing and args.device!='CL':parser.error('GPU weight packing option requires CL')
  if args.precision=='int8' and args.prefill=='off' and args.decode=='off':parser.error('INT8 requires an active NPU phase')
  if args.input_scales and (args.device!='CPU' or args.precision!='int8'):parser.error('Input scaling requires CPU/NPU INT8')
  prompt = [int(value) for value in args.input.read_text().split()]
  if not prompt or args.tokens<2 or args.runs<1 or args.warmups<1 or len(prompt)+args.tokens>args.context:
    parser.error('Need prompt, >=2 output tokens, warmup and measured runs, within context')
  if args.chunk_size is not None and not 1<=args.chunk_size<=256: parser.error('Chunk size must be between 1 and 256')
  if args.warmup_tokens is not None and not 1<=args.warmup_tokens<=len(prompt): parser.error('Warmup length must fit the prompt')
  expected = stock_ids(args.stock_reference,args.tokens) if args.stock_reference else None
  os.environ['DEV'] = args.device
  from tinygrad import Device, Tensor
  from tinygrad.helpers import GlobalCounters
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.kernels.rockchip import configure
  from tinygrad.llm.kernels import rockchip_cpu, rockchip_replay, rockchip_qkv  # noqa: F401 -- load measured sources before hashing

  root = Path(__file__).resolve().parents[2]
  with args.checkpoint.open('rb') as file: checkpoint_hash = hashlib.file_digest(file,'sha256').hexdigest()
  if (expected_hash:=os.environ.get('ONPU_BENCHMARK_CHECKPOINT_SHA256')) and checkpoint_hash!=expected_hash:
    raise ValueError('Checkpoint differs from the benchmark manifest; refusing model loading')
  input_scales = None
  if args.input_scales:
    from extra.rockchip.calibrate_text_int8 import load_scales
    input_scales = load_scales(args.input_scales,checkpoint_hash)
  report = {'complete':False,'device':args.device,'context':args.context,'input_ids':prompt,
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_hash,
            'deferred_weight_realization':args.defer_weight_realization,
            'file_backed_cpu_weights':args.file_backed_cpu_weights,
            'npu_domain_options':{'count':int(os.environ.get('ONPU_NPU_DOMAINS','1')),
                                  'weight_budget_bytes':int(os.environ.get('ONPU_NPU_DOMAIN_BYTES',str(3*1024**3)))},
            'precision':('FP16 stored weights; FP32 projection accumulation; see recorded device placement' if args.precision=='fp16'
                         else 'INT8 per-channel weights and per-token activations; INT32 accumulation, FP32 dequantization and host arithmetic'),
            'sampling':'temperature=0; raw IDs; no template; EOS ignored; empty KV history each request',
            'prefill':args.prefill,'decode':args.decode,'warmups':args.warmups,'runs':args.runs,
            'split_down':args.split_down,'fused_qkv':args.fused_qkv,'cpu_norm':args.cpu_norm,
            'fused_qkv_without_norm':args.fused_qkv_without_norm,
            'fused_swiglu_ffn':args.fused_swiglu_ffn,
            'fused_swiglu_decode':args.fused_swiglu_decode,
            'fused_erf_gelu_ffn':args.fused_erf_gelu_ffn,
            'fused_erf_gelu_decode':args.fused_erf_gelu_decode,
            'swiglu_prefill_rows':int(os.environ.get('ONPU_SWIGLU_PREFILL_ROWS','1')),
            'cpu_static_gemm':args.cpu_static_gemm,
            'cpu_erf_gelu':args.cpu_erf_gelu,
            'cpu_mla_weights':args.cpu_mla_weights,
            'cpu_mla_attention':args.cpu_mla_attention,
            'cpu_mla_projections':args.cpu_mla_projections,
            'cpu_experts':args.cpu_experts,
            'cpu_expert_prefill_rows':args.cpu_expert_prefill_rows,
            'npu_experts':args.npu_experts,
            'npu_expert_activation_residual':args.npu_expert_activation_residual,
            'npu_expert_weight_residual':args.npu_expert_weight_residual,
            'npu_expert_cpu_layers':args.npu_expert_cpu_layers,'npu_expert_cpu_down_layers':args.npu_expert_cpu_down_layers,
            'npu_mla_prefill':args.npu_mla_prefill,'npu_mla_decode':args.npu_mla_decode,
            'npu_mla_precision':args.npu_mla_precision,
            'npu_mla_projections':args.npu_mla_projections,
            'cpu_gelu':args.cpu_gelu,
            'cpu_softcap':args.cpu_softcap,
            'fused_gelu_ffn':args.fused_gelu_ffn,
            'share_projection_weights':args.share_projection_weights,
            'cpu_recurrent_scan':args.cpu_recurrent_scan,
            'gpu_fuse_projections':args.gpu_fuse_projections,'gpu_q8_head':args.gpu_q8_head,
            'gpu_q8_projections':args.gpu_q8_projections,'gpu_pack_weights':not args.gpu_no_weight_packing,
            'input_scales_sha256':hashlib.sha256(args.input_scales.read_bytes()).hexdigest() if args.input_scales else None,
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'initial_clocks':clocks(),'trials':[],
            'source_sha256':source_snapshot(root)}
  report['npu_prefill_attention']=args.npu_prefill_attention
  if args.gpu_q8_projections:report['precision'] += '; native Q8_0 body storage with dequantized operands rounded to FP16'
  def save(): args.report.write_text(json.dumps(report,indent=2)+'\n')
  save()
  begin = time.perf_counter()
  model = model_override
  if model is None:
    model,_ = Transformer.from_gguf(args.checkpoint,max_context=args.context,realize=not args.defer_weight_realization,
                                  mapped_cpu=args.file_backed_cpu_weights)
    Device[args.device].allocator.free_cache()
  elif model.max_context!=args.context: raise ValueError('Loaded model context differs from benchmark')
  gpu_kernels = gpu_kernels_override
  if args.device=='CL' and gpu_kernels is None:
    from tinygrad.llm.kernels.mali import configure as configure_mali
    gpu_kernels = configure_mali(model,fuse_projections=args.gpu_fuse_projections,
                                  q8_head_checkpoint=args.checkpoint if args.gpu_q8_head else None,pack_weights=not args.gpu_no_weight_packing,
                                  recurrent_scan=args.gpu_recurrent_scan,
                                  q8_projections_checkpoint=args.checkpoint if args.gpu_q8_projections else None)
  if gpu_kernels_override is not None:
    if bool(gpu_kernels_override.get('fused_qkv') or gpu_kernels_override.get('fused_gate_up'))!=args.gpu_fuse_projections or \
       bool(gpu_kernels_override.get('q8_output_head'))!=args.gpu_q8_head or \
       bool(gpu_kernels_override.get('q8_projection_layers'))!=args.gpu_q8_projections or \
       gpu_kernels_override.get('pack_weights',True)==args.gpu_no_weight_packing:
      raise ValueError('Reused GPU configuration differs from the requested benchmark flags')
  if args.gpu_q8_projections and (not gpu_kernels or not gpu_kernels.get('q8_projection_layers')):
    raise ValueError('Requested native Q8 body projections did not activate')
  # The hook must be present before chunk selection; otherwise recurrent CPU
  # models still select the old single-token prefill even when it is requested.
  cpu_recurrent_blocks=0
  if args.device=='CPU':
    from tinygrad.llm.kernels.rockchip_recurrent import configure as configure_recurrent
    cpu_recurrent_blocks=configure_recurrent(model,enabled=args.cpu_recurrent_scan)
  chunk_size = args.chunk_size or (gpu_kernels.get('prefill_chunk') if gpu_kernels else None) or len(prompt)
  requested_chunk = chunk_size
  chunk_size = model.prefill_chunk_size(chunk_size)
  report.update(gpu_kernels=gpu_kernels,chunk_size=chunk_size,requested_chunk_size=requested_chunk,
                recurrent_model=model.has_recurrent_block,cpu_recurrent_blocks=cpu_recurrent_blocks,
                warmup_tokens=args.warmup_tokens,model_reused=model_override is not None)
  report['load_seconds'] = time.perf_counter()-begin
  report['gguf_loading'] = model.gguf_loading
  jits = {'prefill':model.greedy_prefill_jit,'decode':model.greedy_rollout_jit}
  begin = time.perf_counter()
  report['placement'] = (configure(model,prefill=args.prefill,decode=args.decode,
                                 prefill_rows=min(chunk_size,256),
                                 tile_columns=args.tile_columns,fuse_projections=args.fuse_projections,
                                 cpu_attention=args.cpu_attention,native_replay=args.native_replay,
                                 npu_prefill_attention=args.npu_prefill_attention,
                                 fused_ffn=args.fused_ffn,split_down=args.split_down,fused_qkv=args.fused_qkv,cpu_norm=args.cpu_norm,
                                 fused_qkv_without_norm=args.fused_qkv_without_norm,
                                 precision=args.precision,cpu_pool=args.cpu_pool,cpu_prefill_pool=args.cpu_prefill_pool,
                                 cpu_fast_softmax=args.cpu_fast_softmax,
                                  cache_cleaning=args.cache_cleaning,wake_helper=args.wake_helper,input_scales=input_scales,
                                  share_projection_weights=args.share_projection_weights,cpu_recurrent_scan=args.cpu_recurrent_scan,
                                  cpu_static_gemm=args.cpu_static_gemm,cpu_gelu=args.cpu_gelu,cpu_softcap=args.cpu_softcap,
                                  cpu_erf_gelu=args.cpu_erf_gelu,
                                  fused_gelu_ffn=args.fused_gelu_ffn,cpu_mla_weights=args.cpu_mla_weights,
                                  fused_swiglu_ffn=args.fused_swiglu_ffn,
                                  fused_swiglu_decode=args.fused_swiglu_decode,
                                  fused_erf_gelu_ffn=args.fused_erf_gelu_ffn,fused_erf_gelu_decode=args.fused_erf_gelu_decode,
                                  cpu_mla_attention=args.cpu_mla_attention,
                                  cpu_mla_projections=args.cpu_mla_projections,
                                  cpu_experts=args.cpu_experts,npu_experts=args.npu_experts,
                                  cpu_expert_prefill_rows=args.cpu_expert_prefill_rows,
                                  npu_expert_cpu_layers=args.npu_expert_cpu_layers,
                                  npu_expert_cpu_down_layers=args.npu_expert_cpu_down_layers,
                                  npu_expert_activation_residual=args.npu_expert_activation_residual,
                                  npu_expert_weight_residual=args.npu_expert_weight_residual,
                                  npu_mla_prefill=args.npu_mla_prefill,npu_mla_decode=args.npu_mla_decode,
                                  npu_mla_precision=args.npu_mla_precision,npu_mla_projections=args.npu_mla_projections)
                         if args.prefill!='off' or args.decode!='off' or args.npu_mla_prefill or args.npu_mla_decode
                           or args.cpu_experts or args.npu_experts or args.cpu_mla_projections or args.cpu_erf_gelu else None)
  if args.npu_experts and not report['placement'].get('npu_expert_blocks',0):
    raise ValueError('Requested NPU experts did not activate for this model')
  if args.fused_swiglu_decode and not report['placement'].get('fused_swiglu_decode_blocks',0):
    raise ValueError('Requested raw SwiGLU decode did not activate for this model')
  if args.fused_erf_gelu_ffn and not report['placement'].get('fused_erf_gelu_ffn_blocks',0):
    raise ValueError('Requested raw erf GELU did not activate for this model')
  if args.fused_erf_gelu_decode and not report['placement'].get('fused_erf_gelu_decode_blocks',0):
    raise ValueError('Requested raw erf GELU decode did not activate for this model')
  if args.cpu_erf_gelu and not report['placement'].get('cpu_erf_gelu_blocks',0):
    raise ValueError('Requested CPU erf GELU did not activate for this model')
  if args.cpu_mla_projections and not report['placement'].get('cpu_mla_projection_blocks',0):
    raise ValueError('Requested CPU MLA projections did not activate for this model')
  if args.cpu_mla_weights and report['placement'] is None:
    from tinygrad.llm.kernels.rockchip_mla import configure_weights
    report['cpu_mla_weights_prepared'] = configure_weights(model)
  else:
    report['cpu_mla_weights_prepared'] = report['placement'].get('cpu_mla_weights',0) if report['placement'] else 0
  if args.cpu_mla_attention and report['placement'] is None:
    from tinygrad.llm.kernels.rockchip_mla_attention import configure as configure_mla_attention
    report['cpu_mla_attention_blocks'] = configure_mla_attention(model)
  else:
    report['cpu_mla_attention_blocks'] = report['placement'].get('cpu_mla_attention_blocks',0) if report['placement'] else 0
  report['cpu_mla_causal_work'] = bool(report['cpu_mla_attention_blocks'] and int(os.environ.get('ONPU_MLA_CAUSAL_WORK','1')))
  report['prepare_seconds'] = time.perf_counter()-begin
  if report['placement'] and report['placement'].get('npu_prefill_attention_blocks',0):
    report['precision'] += '; causal attention uses FP16 Q/K/V and probabilities, FP32 accumulation and CPU FP32 softmax'
  report['grouped_mla_precision'] = report['placement'].get('npu_mla_projection_precision') if report['placement'] else None
  if report['grouped_mla_precision']:
    report['precision'] += ('; grouped MLA uses per-row/per-output INT8, INT32 accumulation and FP32 dequantization'
      if report['grouped_mla_precision'] in ('int8','int8_residual')
      else '; grouped MLA projections use FP16 activations and weights with FP32 accumulation')
    if report['grouped_mla_precision']=='int8_residual':report['precision'] += '; three-product residual correction'
  npu = Device['ROCKCHIP'] if (args.prefill!='off' or args.decode!='off'
    or report['placement'] and report['placement']['npu_mla_projection_blocks']) else None
  report['npu_resident_bytes_after_prepare'] = {device:Device[device].mapped_bytes for device in sorted(Device._opened_devices)
                                               if device.split(':')[0]=='ROCKCHIP'}
  print(json.dumps({'event':'ready','load_seconds':report['load_seconds'],
                    'prepare_seconds':report['prepare_seconds'],'placement':report['placement']}),flush=True)
  save()
  try:
    for index in range(args.warmups+args.runs):
      model._cached_tokens = []
      for block in model.blk:
        if hasattr(block,'cache_kv'): block.cache_kv.assign(Tensor.zeros_like(block.cache_kv)).realize()
      Tensor.manual_seed(0)
      Device[args.device].synchronize()
      run_prompt = prompt[:args.warmup_tokens] if index<args.warmups and args.warmup_tokens else prompt
      row = {'index':index,'warmup':index<args.warmups,'prompt_tokens':len(run_prompt),'clocks_before':clocks()}
      submits_before = npu.submit_count if npu else 0
      ids, stamps = [],[]
      observe('begin',index)
      begin = time.perf_counter()
      for token in itertools.islice(model.generate(run_prompt.copy(),chunk_size=chunk_size,temperature=0),args.tokens):
        stamps.append(time.perf_counter())
        ids.append(token)
        if len(ids)==1:
          first_submits = npu.submit_count if npu else 0
          observe('first',index)
          print(json.dumps({'event':'first_token','index':index,'id':token,'first_token_ms':(stamps[0]-begin)*1000}),flush=True)
      if len(ids)!=args.tokens: raise RuntimeError('Incomplete generated sequence')
      observe('end',index)
      row.update(generated_ids=ids,first_token_ms=(stamps[0]-begin)*1000,
                 prefill_tps=len(run_prompt)/(stamps[0]-begin),decode_tps=(len(ids)-1)/(stamps[-1]-stamps[0]),
                 prefill_submits=first_submits-submits_before,
                 decode_submits=(npu.submit_count if npu else 0)-first_submits,
                 clocks_after=clocks(),matches_stock=ids==expected if expected is not None and run_prompt==prompt else None)
      row.update(allocated_bytes=GlobalCounters.mem_used,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
      report['trials'].append(row)
      report['captured_calls'] = {phase:dict(Counter(f'{call.body.op.name}:{call.device}' for call in jit.captured.linear.src))
                                  for phase,jit in jits.items() if jit.captured is not None}
      report['native_execution'] = {phase:{'runs':getattr(jit.captured,'native_runs',0),'reason':getattr(jit.captured,'reason',None),
        'direct_inputs':len(jit.captured.native.input_copies) if getattr(jit.captured,'native',None) is not None else 0,
        'borrowed_outputs':len(jit.captured.native.output_copies) if getattr(jit.captured,'native',None) is not None else 0,
        'borrow_outputs_enabled':jit.captured.native.borrow_outputs if getattr(jit.captured,'native',None) is not None else False,
        'profile':[{'name':k,'ms':v/1e6,'calls':jit.captured.native.profile_counts[k]}
                   for k,v in jit.captured.native.profile_ns.most_common()] if getattr(jit.captured,'native',None) is not None else []}
                                   for phase,jit in jits.items() if jit.captured is not None}
      save()
      print(json.dumps({'event':'run',**row}),flush=True)
      if row['matches_stock'] is False: raise AssertionError(f'IDs differ from stock: {ids} != {expected}')
    rows = [row for row in report['trials'] if not row['warmup']]
    if args.native_replay:
      assert all(value['runs']>0 for value in report['native_execution'].values()),report['native_execution']
    report.update(complete=True,median_decode_tps=statistics.median(row['decode_tps'] for row in rows),
                  median_first_token_ms=statistics.median(row['first_token_ms'] for row in rows))
    print(json.dumps({key:report[key] for key in ['complete','median_decode_tps','median_first_token_ms']}),flush=True)
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally: save()
  return model,jits


if __name__=='__main__': main()
