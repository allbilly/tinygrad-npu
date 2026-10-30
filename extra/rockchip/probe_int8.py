"""Qualify raw RK3588 INT8 matrices before selecting quantized model execution.

Register/layout reference: https://github.com/oRKLLM/ork-driver
Copyright (c) 2025-2026 Michael Fischer / oRKLLM
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.
THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
import argparse, array, ctypes, json, time
from pathlib import Path
import numpy as np
from tinygrad import Device
from tinygrad.helpers import ceildiv, round_up, to_mv
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import _cmd
from tinygrad.runtime.support.rockchip_int8 import int8_body


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--large-batch',action='store_true')
  args=parser.parse_args()
  dev=Device['ROCKCHIP']
  owned=[]
  def allocate(size,flags=0):
    result=dev.allocate(size,flags)
    owned.append(result)
    return result
  def upload(buf,data):
    ctypes.memmove(buf.addr,data.ctypes.data,data.nbytes)
    dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
  report={'complete':False,'cases':[]}
  rng=np.random.default_rng(20261018)
  try:
    shapes = ([(3,128,1024,3072),(3,128,2048,1024),(3,65,1024,3072)] if args.large_batch else
              [(1,1,512,96),(3,1,1024,3072),(3,17,1024,3072),(3,64,1024,3072),(3,32,3072,1024),(3,1,1024,24576)])
    for cores,m,k,n in shapes:
      width=round_up(ceildiv(n,cores),32)
      parts=[(o,min(width,n-o)) for o in range(0,n,width)]
      if len(parts)!=cores: raise ValueError('Invalid core split')
      a,b,c=allocate(m*k),allocate(n*k),allocate(m*n*4)
      command,task=allocate(4096),allocate(4096,rk.RKNPU_MEM_KERNEL_MAPPING)
      tail=(_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS),_cmd(rk.TARGET_VERSION,0),
            _cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd))
      offset=0
      for i,(start,count) in enumerate(parts):
        # Each core writes a dense private output plane; scatter columns on CPU.
        # ORIG_CHANNEL is not a general destination row-stride control.
        body=int8_body(m,count,k,a.dma,b.dma+start*k,c.dma+start*m*4,count)
        words=array.array('Q',body+tail)
        ctypes.memmove(command.addr+offset*8,words.buffer_info()[0],len(words)*8)
        descriptor=rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,command.dma+offset*8)
        ctypes.memmove(task.addr+i*ctypes.sizeof(descriptor),ctypes.addressof(descriptor),ctypes.sizeof(descriptor))
        offset+=round_up(len(words),2)
      for buf in (command,task): dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      for repeat in range(4):
        x=rng.integers(-128,128,(m,k),dtype=np.int8)
        weight=rng.integers(-128,128,(n,k),dtype=np.int8)
        upload(a,x)
        upload(b,weight.reshape(n//32,32,k//32,32).transpose(0,2,1,3).copy())
        dev.sync(c,rk.RKNPU_MEM_SYNC_TO_DEVICE)
        subcores=[(0,1)] if cores==1 else [(0,0),(0,0),(0,1),(1,1),(2,1)]
        begin=time.perf_counter()
        try:
          rk.DRM_IOCTL_RKNPU_SUBMIT(dev.fd,flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG,
            timeout=6000,task_start=0,task_number=cores,task_counter=0,priority=0,task_obj_addr=task.meta.obj_addr,
            iommu_domain_id=0,task_base_addr=0,hw_elapse_time=0,core_mask=(1<<cores)-1,fence_fd=-1,
            subcore_task=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*s) for s in subcores)))
        except OSError:
          dev.poisoned=True
          raise
        elapsed=time.perf_counter()-begin
        dev.sync(c,rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        raw=np.frombuffer(to_mv(c.addr,m*n*4),np.int32).copy()
        actual=np.concatenate([raw[start*m:(start+count)*m].reshape(m,count) for start,count in parts],axis=1)
        expected=x.astype(np.int32)@weight.astype(np.int32).T
        difference=actual.astype(np.int64)-expected
        row={'shape':[m,k,n],'cores':cores,'repeat':repeat,'submit_ms':elapsed*1000,
             'values':actual.size,'max_abs_error':int(np.abs(difference).max())}
        report['cases'].append(row)
        print(json.dumps(row),flush=True)
        np.testing.assert_array_equal(actual,expected)
    report['complete']=True
  except BaseException as exc:
    report['error']=f'{type(exc).__name__}: {exc}'
    raise
  finally:
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if not dev.poisoned: dev.reset()
    for buf in reversed(owned): dev.free(buf)
    dev.finalize()


if __name__=='__main__': main()
