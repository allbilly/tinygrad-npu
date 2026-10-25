"""Portable operator experiment; it never stops workers or enables serving paths."""
import argparse,hashlib,json,resource,statistics,tempfile,time,uuid
from pathlib import Path
from extra.rockchip.shared_prefill.protocol import (GEOMETRIES,MEMORY_RESERVE_BYTES,SOURCE_DIRECTORY,
                                                  check_memory,sha256 as sha,source_map)
S=SOURCE_DIRECTORY
MODES=('gpu_fp32','npu_copy_fp16','npu_shared_fp16')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--list-cases',action='store_true',help='List geometries without importing device code')
 parser.add_argument('--case',choices=tuple(GEOMETRIES))
 parser.add_argument('--output',type=Path,help='New report path; a sibling .manifest.json records expected sources')
 parser.add_argument('--dry-run',action='store_true',help='Print the plan without importing devices or writing files')
 args=parser.parse_args()
 if args.list_cases:
  print(json.dumps(GEOMETRIES,indent=2));return 0
 if args.case is None or args.output is None:parser.error('--case and --output are required unless --list-cases is used')
 key=args.case;geometry=GEOMETRIES[key];destination=args.output.expanduser().resolve()
 manifest_path=destination.with_suffix('.manifest.json')
 if args.dry_run:
  print(json.dumps({'case':key,'geometry':geometry,'output':str(destination),'manifest':str(manifest_path),
                    'npu_tile_rows':min(geometry['rows'],64,65536//geometry['depth']),
                    'warmup_calls_per_mode':3,'measured_calls_per_mode':9,'memory_reserve_bytes':MEMORY_RESERVE_BYTES,
                    'hardware_access':False,'writes_files':False},indent=2));return 0
 if destination.exists():raise FileExistsError(destination)
 if manifest_path.exists():raise FileExistsError(manifest_path)
 check_memory(startup=True)
 before=source_map()
 expected={'case':key,'geometry':geometry,'source_sha256':before,'run_token':'shared-prefill-'+uuid.uuid4().hex,
           'runner_sha256':sha(Path(__file__)),'helper_sha256':sha(S/'shared_projection.py')}
 destination.parent.mkdir(parents=True,exist_ok=True)
 with manifest_path.open('x') as stream:json.dump(expected,stream,indent=2);stream.write('\n')
 manifest=expected
 rows,columns,depth=(geometry[k] for k in ('rows','columns','depth'))
 report={'schema':'gpu-npu-prefill-sharing-operator-v1','complete':False,'case':key,'geometry':geometry,'source_sha256':before,
         'run_token':manifest['run_token'],'runner_sha256':sha(Path(__file__)),'helper_sha256':sha(S/'shared_projection.py'),
         'scope':'Synthetic GPU-resident projection only. NPU FP16 rounds FP32 activations to FP16; no model-quality or serving/speed qualification.',
         'model_qualified':False,'API_qualified':False,'quality_qualified':False,'speed_qualified':False,'warmups':[],'trials':[],'guards':[],'events':[]}
 report['expected_manifest']={'file':manifest_path.name,'sha256':sha(manifest_path)}
 report['memory_reserve_bytes']=MEMORY_RESERVE_BYTES
 def save():
  temporary=None
  try:
   with tempfile.NamedTemporaryFile(mode='w',prefix='.'+destination.name+'.',dir=destination.parent,delete=False) as stream:
    temporary=Path(stream.name);json.dump(report,stream,indent=2);stream.write('\n')
   temporary.replace(destination)
  finally:
   if temporary is not None:temporary.unlink(missing_ok=True)
 def event(name,**kw):
  row={'event':name,**kw};report['events'].append(row);save();print(json.dumps(row),flush=True)
 from tinygrad import Device,Tensor,TinyJit
 from tinygrad.llm.kernels.mali_linear import gemm,pack_weight as gpu_pack
 from tinygrad.llm.kernels.rockchip import pack_weight as npu_pack,packed_matmul
 from extra.rockchip.shared_prefill.shared_projection import SharedProjection
 import numpy as np
 gpu=None;npu=None;shared=None
 def readback(t):
  out=np.empty(t.shape,dtype=np.float32)
  gpu.allocator._copyout(memoryview(out).cast('B'),t.uop.buffer._buf)
  return out
 try:
  gpu=Device['CL'];npu=Device['ROCKCHIP']
  report['device']={'name':gpu.device_name,'driver':gpu.driver_version,'extensions':gpu.device_exts}
  if 'cl_arm_import_memory_dma_buf' not in gpu.device_exts:raise RuntimeError('Required GPU DMA-BUF import extension unavailable')
  rng=np.random.default_rng(1031+columns+depth)
  weights=rng.normal(0,.02,(columns,depth)).astype(np.float16)
  arrays=[rng.normal(0,.2,(rows,depth)).astype(np.float32) for _ in range(3)]
  report['data_sha256']={'weight':hashlib.sha256(weights.tobytes()).hexdigest(),'inputs':[hashlib.sha256(x.tobytes()).hexdigest() for x in arrays]}
  reference_fp32=[a@weights.astype(np.float32).T for a in arrays]
  reference_fp16=[a.astype(np.float16).astype(np.float32)@weights.astype(np.float32).T for a in arrays]
  xs=[Tensor(a,device='CL').realize() for a in arrays]
  wg=Tensor(weights,device='CL').realize();pg=gpu_pack(wg).realize()
  wn=npu_pack(Tensor(weights,device='CPU')).realize()
  tile=min(rows,64,65536//depth);report['npu_tile_rows']=tile;report['npu_tiles_per_call']=rows//tile
  shared=SharedProjection(rows,columns,depth,wn)
  gf=TinyJit(lambda x:gemm(x,wg,pg).realize())
  def copy(x):return Tensor.cat(*(packed_matmul(x[start:start+tile],wn,columns) for start in range(0,rows,tile)),dim=0).realize()
  cf=TinyJit(copy)
  functions={'gpu_fp32':gf,'npu_copy_fp16':cf,'npu_shared_fp16':shared}
  copy_refs={}
  def run(mode,index,phase,round_number):
   check_memory()
   gpu.synchronize();prior=npu.submit_count;alloc=npu.allocation_count;began=time.perf_counter()
   output=functions[mode](xs[index]);gpu.synchronize();elapsed=time.perf_counter()-began
   measured_submissions=npu.submit_count-prior;actual=readback(output)
   reference=reference_fp32[index] if mode=='gpu_fp32' else reference_fp16[index]
   difference=np.abs(actual-reference);tolerance=2e-4+3e-4*np.abs(reference)
   good=bool(np.isfinite(actual).all() and np.all(difference<=tolerance))
   half_delta=np.abs(actual-reference_fp32[index])
   bit_equal=None
   if mode=='npu_copy_fp16' and phase=='warmup':copy_refs[index]=actual.copy()
   elif mode!='gpu_fp32':bit_equal=bool(np.array_equal(actual.view(np.uint32),copy_refs[index].view(np.uint32)))
   row={'mode':mode,'phase':phase,'round':round_number,'input_index':index,'seconds':elapsed,
        'npu_submissions':measured_submissions,'npu_allocations_during_call':npu.allocation_count-alloc,
        'finite':bool(np.isfinite(actual).all()),'reference_kind':'FP32_input_FP16_weight' if mode=='gpu_fp32' else 'FP16_rounded_input_FP16_weight',
        'reference_allclose':good,'max_abs_error':float(difference.max()),'max_tolerance_ratio':float((difference/tolerance).max()),
        'max_abs_error_against_FP32_input':float(half_delta.max()),'relative_RMS_against_FP32_input':float(np.linalg.norm(half_delta)/max(np.linalg.norm(reference_fp32[index]),1e-30)),
        'bit_equal_to_copy':bit_equal,'output_sha256':hashlib.sha256(actual.tobytes()).hexdigest()}
   (report['warmups'] if phase=='warmup' else report['trials']).append(row);save()
   if not good or (mode!='gpu_fp32' and phase=='measured' and bit_equal is not True):raise ValueError('Numerical parity failure in '+mode)
   if mode=='npu_shared_fp16' and bit_equal is not True:raise ValueError('Shared/copy FP16 outputs differ')
   if measured_submissions!=(0 if mode=='gpu_fp32' else rows//tile):raise ValueError('Unexpected NPU submission count')
   return row
  event('warmup_started',geometry=geometry)
  # Warm the existing-copy path before shared comparisons, for all three inputs.
  for mode in MODES:
   for i in range(3):run(mode,i,'warmup',i)
  event('measured_started')
  for i in range(9):
   order=MODES[i%3:]+MODES[:i%3]
   for mode in order:run(mode,i%3,'measured',i)
  # Invalid first-tile values must fail before a hardware submission, and the
  # same object must subsequently produce a correct valid result.
  for value in (float('nan'),float('inf'),1e10):
   bad=arrays[0].copy();bad[0,0]=value;x=Tensor(bad,device='CL').realize();gpu.synchronize();prior=npu.submit_count
   error=None
   try:shared(x)
   except ValueError as caught:error=str(caught)
   row={'kind':'invalid_activation','value':str(value),'rejected':error is not None,'error':error,'npu_submissions':npu.submit_count-prior}
   report['guards'].append(row);save()
   if error is None or row['npu_submissions']!=0:raise ValueError('Invalid activation guard failed')
  bigger=Tensor(np.concatenate((arrays[0][:1],arrays[0]),axis=0),device='CL').realize()
  prior=npu.submit_count;rejected=False
  try:shared(bigger[1:])
  except ValueError:rejected=True
  report['guards'].append({'kind':'offset_input_view','rejected':rejected,'npu_submissions':npu.submit_count-prior})
  if not rejected or npu.submit_count!=prior:raise ValueError('Offset input view guard failed')
  packed_array=weights.reshape(columns//16,16,depth//32,32).transpose(0,2,1,3).copy()
  larger_weight=Tensor(np.concatenate((packed_array[:1],packed_array),axis=0),device='CPU').to('ROCKCHIP').realize()
  prior=npu.submit_count;alloc=npu.allocation_count;rejected=False
  try:SharedProjection(rows,columns,depth,larger_weight[1:])
  except ValueError:rejected=True
  report['guards'].append({'kind':'offset_weight_view','rejected':rejected,'npu_submissions':npu.submit_count-prior,
                           'npu_allocations_in_constructor':npu.allocation_count-alloc})
  if not rejected or npu.submit_count!=prior or npu.allocation_count!=alloc:raise ValueError('Offset weight view guard failed')
  recovery=readback(shared(xs[2]))
  report['recovery_bit_equal_to_copy']=bool(np.array_equal(recovery.view(np.uint32),copy_refs[2].view(np.uint32)))
  if not report['recovery_bit_equal_to_copy']:raise ValueError('Post-guard recovery differs')
  report['summaries']={mode:{'median_seconds':statistics.median(row['seconds'] for row in report['trials'] if row['mode']==mode),
                             'measured_runs':sum(row['mode']==mode for row in report['trials'])} for mode in MODES}
  report['source_sha256_after']=source_map()
  report['process_lifetime_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  report['complete']=report['source_sha256_after']==before
  event('complete',complete=report['complete'],summaries=report['summaries'])
 except BaseException as error:
  report['error']={'type':type(error).__name__,'message':str(error)};raise
 finally:
  if shared is not None:shared.close()
  report.setdefault('source_sha256_after',source_map());save()
 return 0 if report['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
