"""Experimentally qualify 64-channel K alignment without changing backend guards.

Derive the same register recipe as support.rockchip_int8, conservatively retaining
the CBUF allocation of the next 512-channel shape. Only K-addressing fields change.
No model selects these shapes until exact integer golden checks pass.
"""
import argparse, array, ctypes, json, time
from pathlib import Path
import numpy as np
from tinygrad import Device
from tinygrad.helpers import ceildiv, round_up, to_mv
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import _cmd
from tinygrad.runtime.support.rockchip_int8 import int8_body

def natural_body(m,n,k,lhs,rhs,dst):
  body = int8_body(m,n,round_up(k,512),lhs,rhs,dst,n)
  values = {0x1024:((k-1)<<16)|k,0x1030:k*n,0x1034:k,0x1044:ceildiv(k,64),0x107c:k//16,0x1088:k}
  return tuple(_cmd(0x201,word&0xffff,values[word&0xffff]) if word>>48==0x201 and word&0xffff in values else word for word in body)

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  dev = Device['ROCKCHIP']
  owned = []
  def allocate(size,flags=0):
    buf = dev.allocate(size,flags)
    owned.append(buf)
    return buf
  def upload(buf,values):
    ctypes.memmove(buf.addr,values.ctypes.data,values.nbytes)
    dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
  report = {'complete':False,'cases':[],'method':'stock next-512 CBUF allocation; actual K addressing; no production guards changed'}
  rng = np.random.default_rng(20261019)
  try:
    for m,k,n in [(1,576,576),(64,576,576),(128,768,3072),(32,1152,576),(1,576,24576),(1,576,128)]:
      width = round_up(ceildiv(n,3),32)
      parts = [(start,min(width,n-start)) for start in range(0,n,width)]
      if len(parts)!=3: parts = [(0,n)]
      cores = len(parts)
      a,b,c = allocate(m*k),allocate(n*k),allocate(m*n*4)
      command,task = allocate(8192),allocate(4096,rk.RKNPU_MEM_KERNEL_MAPPING)
      tail = (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS),_cmd(rk.TARGET_VERSION,0),
              _cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd))
      offset = 0
      for i,(start,count) in enumerate(parts):
        body = natural_body(m,count,k,a.dma,b.dma+start*k,c.dma+start*m*4)
        words = array.array('Q',body+tail)
        ctypes.memmove(command.addr+offset*8,words.buffer_info()[0],len(words)*8)
        descriptor = rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,command.dma+offset*8)
        ctypes.memmove(task.addr+i*ctypes.sizeof(descriptor),ctypes.addressof(descriptor),ctypes.sizeof(descriptor))
        offset += round_up(len(words),2)
      for buf in (command,task): dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      for repeat in range(2):
        x = rng.integers(-128,128,(m,k),dtype=np.int8)
        weight = rng.integers(-128,128,(n,k),dtype=np.int8)
        upload(a,x)
        upload(b,weight.reshape(n//32,32,k//32,32).transpose(0,2,1,3).copy())
        dev.sync(c,rk.RKNPU_MEM_SYNC_TO_DEVICE)
        subcores = [(0,1)] if cores==1 else [(0,0),(0,0),(0,1),(1,1),(2,1)]
        begin = time.perf_counter()
        try:
          rk.DRM_IOCTL_RKNPU_SUBMIT(dev.fd,flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG,
            timeout=6000,task_start=0,task_number=cores,task_counter=0,priority=0,task_obj_addr=task.meta.obj_addr,
            regcfg_obj_addr=0,task_base_addr=0,user_data=0,core_mask=(1<<cores)-1,fence_fd=-1,
            subcore_task=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*s) for s in subcores)))
        except OSError:
          dev.poisoned = True
          raise
        elapsed = time.perf_counter()-begin
        dev.sync(c,rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        raw = np.frombuffer(to_mv(c.addr,m*n*4),np.int32).copy()
        actual = np.concatenate([raw[start*m:(start+count)*m].reshape(m,count) for start,count in parts],axis=1)
        expected = x.astype(np.int32) @ weight.astype(np.int32).T
        error = int(abs(actual.astype(np.int64)-expected).max())
        row = {'shape':[m,k,n],'cores':cores,'repeat':repeat,'submit_ms':elapsed*1000,'values':actual.size,'max_abs_error':error}
        report['cases'].append(row)
        print(json.dumps(row),flush=True)
        np.testing.assert_array_equal(actual,expected)
    report['complete'] = True
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally:
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if not dev.poisoned: dev.reset()
    for buf in reversed(owned): dev.free(buf)
    dev.finalize()

if __name__ == '__main__': main()
