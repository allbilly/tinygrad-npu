"""Strict stdlib-only validation for the experimental GPU/NPU operator report."""
import math
import statistics

SCHEMA = 'gpu-npu-prefill-sharing-operator-v1'
MODES = ('gpu_fp32', 'npu_copy_fp16', 'npu_shared_fp16')
GUARD_VALUES = ('nan', 'inf', '10000000000.0')


def _mapping(value): return isinstance(value, dict)


def _integer(value, minimum=0): return type(value) is int and value >= minimum


def _finite(value, *, positive=False):
  return (type(value) in (int, float) and math.isfinite(value) and
          (value > 0 if positive else True))


def _digest(value):
  return (isinstance(value, str) and len(value) == 64 and
          all(ch in '0123456789abcdef' for ch in value))


def validate(raw, expected):
  """Validate one raw report against the manifest-derived expected entry."""
  errors = []

  def require(ok, message):
    if not ok: errors.append(message)

  if not _mapping(raw): return {'valid':False, 'errors':['raw must be an object']}
  if not _mapping(expected): return {'valid':False, 'errors':['expected must be an object']}

  require(raw.get('schema') == SCHEMA, 'schema must identify this operator probe')
  require(raw.get('complete') is True, 'complete must be true')
  require('error' not in raw, 'raw report must not contain an error')

  for key in ('case', 'geometry', 'run_token', 'runner_sha256', 'helper_sha256', 'source_sha256'):
    require(key in expected, 'expected.' + key + ' is required')
    require(raw.get(key) == expected.get(key), key + ' differs from expected')
  require(isinstance(expected.get('case'), str) and bool(expected.get('case')),
          'expected.case must be a nonempty string')
  geometry = expected.get('geometry')
  require(_mapping(geometry), 'expected.geometry must be an object')
  if _mapping(geometry):
    require(set(geometry) == {'rows', 'columns', 'depth'},
            'expected.geometry must contain exactly rows, columns, and depth')
    for key in ('rows', 'columns', 'depth'):
      require(_integer(geometry.get(key), 1), 'expected.geometry.' + key + ' must be a positive integer')
  require(isinstance(expected.get('run_token'), str) and bool(expected.get('run_token')),
          'expected.run_token must be a nonempty string')
  for key in ('runner_sha256', 'helper_sha256'):
    require(_digest(expected.get(key)), 'expected.' + key + ' must be a SHA-256 digest')

  wanted_sources = expected.get('source_sha256')
  require(_mapping(wanted_sources) and bool(wanted_sources), 'expected.source_sha256 must be a nonempty object')
  if _mapping(wanted_sources):
    require(all(isinstance(path, str) and bool(path) and _digest(digest)
                for path, digest in wanted_sources.items()),
            'expected.source_sha256 must map paths to SHA-256 digests')
  require(raw.get('source_sha256') == wanted_sources, 'source_sha256 differs from expected')
  require(raw.get('source_sha256_after') == wanted_sources, 'source_sha256_after differs from expected')

  for key in ('model_qualified', 'API_qualified', 'quality_qualified', 'speed_qualified'):
    require(raw.get(key) is False, key + ' must remain false')
  require(isinstance(raw.get('scope'), str) and bool(raw.get('scope')),
          'scope must describe the limited experiment')

  rows = geometry.get('rows') if _mapping(geometry) else None
  depth = geometry.get('depth') if _mapping(geometry) else None
  tile = min(rows, 64, 65536 // depth) if all(_integer(v, 1) for v in (rows, depth)) else 0
  require(tile > 0, 'geometry must produce a positive NPU tile')
  if tile:
    require(rows % tile == 0, 'rows must be divisible by the NPU tile')
    require(raw.get('npu_tile_rows') == tile, 'npu_tile_rows differs from geometry')
    require(raw.get('npu_tiles_per_call') == rows // tile, 'npu_tiles_per_call differs from geometry')

  device = raw.get('device')
  require(_mapping(device), 'device must be an object')
  extensions = device.get('extensions') if _mapping(device) else None
  require(isinstance(extensions, (list, tuple)) and 'cl_arm_import_memory_dma_buf' in extensions,
          'device must report cl_arm_import_memory_dma_buf')

  data = raw.get('data_sha256')
  require(_mapping(data), 'data_sha256 must be an object')
  if _mapping(data):
    inputs = data.get('inputs')
    require(_digest(data.get('weight')), 'data_sha256.weight must be a SHA-256 digest')
    require(isinstance(inputs, list) and len(inputs) == 3 and all(_digest(x) for x in inputs),
            'data_sha256.inputs must contain three SHA-256 digests')
    if isinstance(inputs, list) and len(inputs) == 3 and all(_digest(x) for x in inputs):
      require(len(set(inputs)) == 3, 'the three input hashes must be unique')

  def check_measurement(row, *, phase, mode, index, round_number):
    prefix = phase + '[' + str(round_number) + '].' + mode
    require(_mapping(row), prefix + ' must be an object')
    if not _mapping(row): return
    require(row.get('mode') == mode, prefix + '.mode mismatch')
    require(row.get('phase') == phase, prefix + '.phase mismatch')
    require(row.get('input_index') == index, prefix + '.input_index mismatch')
    require(row.get('round') == round_number, prefix + '.round mismatch')
    require(_finite(row.get('seconds'), positive=True), prefix + '.seconds must be finite and positive')
    require(row.get('finite') is True, prefix + '.finite must be true')
    require(row.get('reference_allclose') is True, prefix + '.reference_allclose must be true')
    require(_finite(row.get('max_tolerance_ratio')) and 0 <= row['max_tolerance_ratio'] <= 1,
            prefix + '.max_tolerance_ratio must be finite and at most one')
    require(_finite(row.get('max_abs_error')) and row['max_abs_error'] >= 0,
            prefix + '.max_abs_error must be finite and nonnegative')
    require(_finite(row.get('relative_RMS_against_FP32_input')) and row['relative_RMS_against_FP32_input'] >= 0,
            prefix + '.relative_RMS_against_FP32_input must be finite and nonnegative')
    require(_finite(row.get('max_abs_error_against_FP32_input')) and row['max_abs_error_against_FP32_input'] >= 0,
            prefix + '.max_abs_error_against_FP32_input must be finite and nonnegative')
    require(_digest(row.get('output_sha256')), prefix + '.output_sha256 must be a SHA-256 digest')
    expected_ref = 'FP32_input_FP16_weight' if mode == 'gpu_fp32' else 'FP16_rounded_input_FP16_weight'
    require(row.get('reference_kind') == expected_ref, prefix + '.reference_kind mismatch')
    expected_submissions = 0 if mode == 'gpu_fp32' else (rows // tile if tile else None)
    require(row.get('npu_submissions') == expected_submissions, prefix + '.npu_submissions mismatch')
    bit_equal = row.get('bit_equal_to_copy')
    if mode == 'gpu_fp32': require(bit_equal is None, prefix + '.bit_equal_to_copy must be null for GPU mode')
    elif mode == 'npu_shared_fp16' or phase == 'measured':
      require(bit_equal is True, prefix + '.bit_equal_to_copy must be true')
    elif phase == 'warmup':
      require(bit_equal is None or bit_equal is True, prefix + '.bit_equal_to_copy has invalid warmup value')
    if phase == 'measured' and mode == 'npu_shared_fp16':
      require(row.get('npu_allocations_during_call') == 0,
              prefix + '.npu_allocations_during_call must be zero after warmup')
    if phase == 'measured':
      require(_integer(row.get('npu_allocations_during_call')),
              prefix + '.npu_allocations_during_call must be a nonnegative integer')

  warmups = raw.get('warmups')
  require(isinstance(warmups, list) and len(warmups) == 9, 'warmups must contain exactly nine rows')
  if isinstance(warmups, list) and len(warmups) == 9:
    expected_warm = [(mode, index) for mode in MODES for index in range(3)]
    for row, (mode, index) in zip(warmups, expected_warm):
      check_measurement(row, phase='warmup', mode=mode, index=index, round_number=index)

  trials = raw.get('trials')
  require(isinstance(trials, list) and len(trials) == 27, 'trials must contain exactly 27 rows')
  if isinstance(trials, list) and len(trials) == 27:
    expected_trials = [(mode, index % 3, index) for index in range(9)
                       for mode in MODES[index % 3:] + MODES[:index % 3]]
    for row, (mode, input_index, round_number) in zip(trials, expected_trials):
      check_measurement(row, phase='measured', mode=mode, index=input_index, round_number=round_number)

  guards = raw.get('guards')
  require(isinstance(guards, list) and len(guards) == 5, 'guards must contain exactly five rows')
  if isinstance(guards, list) and len(guards) == 5:
    for i, value in enumerate(GUARD_VALUES):
      guard = guards[i]
      require(_mapping(guard), 'guards[' + str(i) + '] must be an object')
      if _mapping(guard):
        require(guard.get('kind') == 'invalid_activation', 'guards[' + str(i) + '] must check invalid activations')
        require(guard.get('value') == value, 'guards[' + str(i) + '].value mismatch')
        require(guard.get('rejected') is True, 'guards[' + str(i) + '] must be rejected')
        require(guard.get('npu_submissions') == 0, 'guards[' + str(i) + '] must not submit to NPU')
    for i, kind in ((3, 'offset_input_view'), (4, 'offset_weight_view')):
      guard = guards[i]
      require(_mapping(guard), 'guards[' + str(i) + '] must be an object')
      if _mapping(guard):
        require(guard.get('kind') == kind, 'guards[' + str(i) + '].kind mismatch')
        require(guard.get('rejected') is True, 'guards[' + str(i) + '] must be rejected')
        require(guard.get('npu_submissions') == 0, 'guards[' + str(i) + '] must not submit to NPU')
        if kind == 'offset_weight_view':
          require(guard.get('npu_allocations_in_constructor') == 0,
                  'offset weight guard must allocate no NPU buffers')

  require(raw.get('recovery_bit_equal_to_copy') is True,
          'recovery_bit_equal_to_copy must be true after guard rejections')

  summaries = raw.get('summaries')
  require(_mapping(summaries), 'summaries must be an object')
  if _mapping(summaries):
    require(set(summaries) == set(MODES), 'summaries must contain exactly the three modes')
    if isinstance(trials, list):
      for mode in MODES:
        summary = summaries.get(mode)
        require(_mapping(summary), 'summaries.' + mode + ' must be an object')
        if not _mapping(summary): continue
        mode_rows = [row for row in trials if _mapping(row) and row.get('mode') == mode]
        require(len(mode_rows) == 9, 'trials must contain nine measured rows for ' + mode)
        require(summary.get('measured_runs') == len(mode_rows), 'summaries.' + mode + '.measured_runs mismatch')
        if mode_rows and all(_finite(row.get('seconds'), positive=True) for row in mode_rows):
          median = statistics.median(row['seconds'] for row in mode_rows)
          recorded = summary.get('median_seconds')
          require(_finite(recorded, positive=True) and math.isclose(recorded, median, rel_tol=1e-12, abs_tol=0.0),
                  'summaries.' + mode + '.median_seconds mismatch')

  return {'valid':not errors, 'errors':errors}


def main():
  """Validate the report and its pinned manifest without opening a device."""
  import argparse
  import json
  from pathlib import Path
  from extra.rockchip.shared_prefill.protocol import GEOMETRIES, SOURCE_ROOT, sha256, source_map

  parser = argparse.ArgumentParser(description=main.__doc__)
  inputs = parser.add_mutually_exclusive_group(required=True)
  inputs.add_argument('--report', type=Path)
  inputs.add_argument('--recorded', action='store_true', help='Check the packaged historical measurements without devices')
  parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT,
                      help='Checkout to compare with the recorded source hashes')
  args = parser.parse_args()
  if args.recorded:
    import hashlib
    fixture = json.loads((Path(__file__).resolve().parent / 'recorded-results.json').read_text())
    errors = []
    if fixture.get('schema') != 'shared-prefill-recorded-results-v1' or fixture.get('historical') is not True:
      errors.append('Packaged records must be labeled as historical')
    if fixture.get('portable_cli_rerun') is not False:
      errors.append('Historical measurements cannot claim a portable CLI rerun')
    if sha256(Path(__file__).resolve().parent / 'shared_projection.py') != fixture.get('measured_helper_sha256'):
      errors.append('Packaged helper differs from the measured helper')
    records = fixture.get('cases', [])
    if len(records) != len(GEOMETRIES) or {r['report']['case'] for r in records} != set(GEOMETRIES):
      errors.append('Packaged records must cover the four measured geometries')
    for record in records:
      raw, expected = record['report'], record['expected']
      canonical = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
      if canonical != record.get('record_canonical_sha256'):
        errors.append(raw.get('case', '?') + ': packaged record changed')
      if expected.get('geometry') != GEOMETRIES.get(raw.get('case')):
        errors.append(raw.get('case', '?') + ': unexpected geometry')
      errors.extend(raw.get('case', '?') + ': ' + message for message in validate(raw, expected)['errors'])
    print(json.dumps({'valid': not errors, 'historical': True, 'portable_cli_rerun': False, 'cases': len(records), 'errors': errors}, indent=2))
    return 0 if not errors else 1
  report_path = args.report.expanduser().resolve()
  raw = json.loads(report_path.read_text())
  identity = raw.get('expected_manifest', {})
  filename = identity.get('file')
  if not isinstance(filename, str) or Path(filename).name != filename or filename in ('.', '..'):
    raise ValueError('The report must name a sibling manifest file')
  manifest_path = report_path.parent / filename
  if sha256(manifest_path) != identity.get('sha256'):
    raise ValueError('Expected-manifest hash mismatch')
  expected = json.loads(manifest_path.read_text())
  if expected.get('geometry') != GEOMETRIES.get(expected.get('case')):
    raise ValueError('Expected case is not a supported geometry')
  source_root = args.source_root.expanduser().resolve()
  if source_map(source_root) != expected.get('source_sha256'):
    raise ValueError('Current checkout differs from the expected source cohort')
  module_root = source_root / 'extra/rockchip/shared_prefill'
  if (sha256(module_root / 'benchmark.py') != expected.get('runner_sha256')
      or sha256(module_root / 'shared_projection.py') != expected.get('helper_sha256')):
    raise ValueError('Runner or helper source differs from the expected manifest')
  verdict = validate(raw, expected)
  print(json.dumps(verdict, indent=2))
  return 0 if verdict['valid'] else 1


if __name__ == '__main__':
  raise SystemExit(main())
