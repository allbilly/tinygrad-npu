"""Standard-library option helpers for explicit CPU/NPU quality configurations."""


def add_projection_options(parser):
  parser.add_argument('--share-projection-weights', action='store_true',
                      help='Match the opt-in shared packed INT8 projection path')
  parser.add_argument('--cpu-prefill-pool', action='store_true',
                      help='Use the CPU worker pool during prefill as well as decode')
  parser.add_argument('--cpu-recurrent-scan', action='store_true',
                      help='Match the experimental CPU GatedDeltaNet scan path')
  return parser


def projection_options(args):
  return {"share_projection_weights": bool(args.share_projection_weights),
          "cpu_prefill_pool": bool(args.cpu_prefill_pool),
          "cpu_recurrent_scan": bool(args.cpu_recurrent_scan)}


def projection_option_error(*, mode, precision, device, checkpoint_is_text, share_projection_weights, cpu_prefill_pool,
                            cpu_recurrent_scan=False):
  if not (share_projection_weights or cpu_prefill_pool or cpu_recurrent_scan): return None
  if mode != 'candidate': return 'Projection sharing and CPU prefill pooling are candidate-only options'
  if precision != 'int8' or device != 'CPU' or not checkpoint_is_text:
    return 'Projection sharing and CPU prefill pooling require a text GGUF CPU+NPU INT8 candidate'
  return None


def fp16_reference_error(reference, candidate, manifest):
  """Require a matched local FP16 reference; it never certifies RKLLM parity."""
  if (reference.get('device')!='CL' or reference.get('precision')!='fp16' or reference.get('gpu_npu') is not False
      or reference.get('gpu_q8_projections') is not False):
    return 'Local FP16 reference requires a GPU-only FP16 body'
  if candidate.get('device')!='CPU' or candidate.get('precision')!='int8':
    return 'Local FP16 comparison requires the CPU+NPU INT8 candidate'
  placement=candidate.get('placement')
  if (not isinstance(placement,dict) or placement.get('precision')!='int8'
      or any(type(placement.get(key)) is not int or placement[key]<=0 for key in ('layers','prefill_layers','decode_layers'))):
    return 'Local FP16 comparison requires positive INT8 NPU placement in both phases'
  expected=manifest.get('checkpoint_sha256')
  if not isinstance(expected,str) or len(expected)!=64 or any(r.get('checkpoint_sha256')!=expected for r in (reference,candidate)):
    return 'Local reference and candidate must match the manifest checkpoint hash'
  hashes=reference.get('source_sha256')
  if not isinstance(hashes,dict) or not hashes or hashes!=candidate.get('source_sha256'):
    return 'Local reference and candidate must use the same captured source cohort'
  return None
