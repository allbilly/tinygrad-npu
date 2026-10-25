"""Qualify chained, independent FP16 attention-sized matrices on the NPU.

Uses the retained FP16 register recipe. PC next amounts follow the verified GEMM
chain convention: ceil(next body qwords / 2) + 1. Each core has its own chain.
"""
import argparse, array, ctypes, json, time
from pathlib import Path
import numpy as np
from tinygrad import Device
from tinygrad.helpers import ceildiv, round_up, to_mv
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import _cmd, _matmul_body

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
  def upload(buf,data):
    ctypes.memmove(buf.addr,data.ctypes.data,data.nbytes)
    dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
  report = {'complete':False,'cases':[]}
  rng = np.random.default_rng(20261019)
  try:
    for cores,rows,k,n in [(1,128,64,128),(1,1024,64,1024),(3,1024,64,1024),(3,1024,1024,64)]:
      tile = 64
      tiles = rows//tile
      a,b,c = allocate(cores*rows*k*2),allocate(cores*n*k*2),allocate(cores*rows*n*4)
      bodies = [_matmul_body(tile,n,k,a.dma+(head*rows+first)*k*2,b.dma+head*n*k*2,
                            c.dma+(head*rows+first)*n*4) for head in range(cores) for first in range(0,rows,tile)]
      stride = round_up(len(bodies[0])+4,2)
      command = allocate(len(bodies)*stride*8)
      tasks = allocate(len(bodies)*ctypes.sizeof(rk.struct_rknpu_task),rk.RKNPU_MEM_KERNEL_MAPPING)
      for i,body in enumerate(bodies):
        next_task = i+1 if (i+1)%tiles else None
        tail = ((_cmd(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,(command.dma+next_task*stride*8)&0xfffffff0),
                 _cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,ceildiv(len(bodies[next_task]),2)+1)) if next_task is not None else
                (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS)))
        words = array.array('Q',body+tail+(_cmd(rk.TARGET_VERSION,0),_cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd)))
        ctypes.memmove(command.addr+i*stride*8,words.buffer_info()[0],len(words)*8)
        desc = rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,command.dma+i*stride*8)
        ctypes.memmove(tasks.addr+i*ctypes.sizeof(desc),ctypes.addressof(desc),ctypes.sizeof(desc))
      for buf in (command,tasks): dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      subcores = [(0,tiles)] if cores==1 else [(0,0),(0,0),(0,tiles),(tiles,tiles),(2*tiles,tiles)]
      for repeat in range(3):
        x = rng.normal(0,.2,(cores,rows,k)).astype(np.float16)
        weight = rng.normal(0,.2,(cores,n,k)).astype(np.float16)
        upload(a,x.reshape(cores,tiles,tile,k//8,8).transpose(0,1,3,2,4).copy())
        upload(b,weight.reshape(cores,n//16,16,k//32,32).transpose(0,1,3,2,4).copy())
        dev.sync(c,rk.RKNPU_MEM_SYNC_TO_DEVICE)
        start = time.perf_counter()
        try:
          rk.DRM_IOCTL_RKNPU_SUBMIT(dev.fd,flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG,timeout=6000,
            task_start=0,task_number=len(bodies),task_counter=0,priority=0,task_obj_addr=tasks.meta.obj_addr,
            regcfg_obj_addr=0,task_base_addr=0,user_data=0,core_mask=(1<<cores)-1,fence_fd=-1,
            subcore_task=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*part) for part in subcores)))
        except OSError:
          dev.poisoned = True
          raise
        elapsed = time.perf_counter()-start
        dev.sync(c,rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        raw = np.frombuffer(to_mv(c.addr,cores*rows*n*4),np.float32).copy()
        actual = raw.reshape(cores,tiles,n//4,tile,4).transpose(0,1,3,2,4).reshape(cores,rows,n)
        expected = x.astype(np.float32) @ weight.astype(np.float32).swapaxes(-1,-2)
        row = {'cores':cores,'shape':[rows,k,n],'tasks':len(bodies),'repeat':repeat,'submit_ms':elapsed*1000,
               'values':actual.size,'max_abs':float(abs(actual-expected).max()),'finite':bool(np.isfinite(actual).all())}
        report['cases'].append(row)
        print(json.dumps(row),flush=True)
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
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
