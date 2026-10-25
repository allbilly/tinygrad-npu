"""Check whether complete captured register initialization permits reset-free matmul reuse.

This diagnostic reads an existing stock register capture, substitutes every live
geometry/address field, and checks complete outputs against FP32 NumPy products.
It does not change the production submission policy.
"""
import argparse, array, ctypes, hashlib, json, time
import numpy as np
from tinygrad import Device
from tinygrad.helpers import ceildiv, round_up, to_mv
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import _cmd, _matmul_body


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--capture',required=True)
  parser.add_argument('--report',required=True)
  parser.add_argument('--cores',type=int,choices=[1,3],default=1)
  args = parser.parse_args()
  capture = open(args.capture,'rb').read()
  template = json.loads(capture)['tasks'][0]['registers']
  dev = Device['ROCKCHIP']
  rng = np.random.default_rng(20261018)
  report = {'complete':False,'capture_sha256':hashlib.sha256(capture).hexdigest(),'cases':[]}
  geometry = _matmul_body
  owned = []
  def alloc(size,flags=0):
    buf = dev.allocate(size,flags)
    owned.append(buf)
    return buf
  def copy(buf,data):
    data = data.tobytes()
    ctypes.memmove(buf.addr,data,len(data))
    dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
  try:
    # Alternate batched/decode geometries and inputs, preserving the complete K contraction.
    shapes = [(1,1024,1024),(16,1024,1024),(64,1024,1024),(1,1024,1024),(8,3072,1024)]
    for m,k,n in shapes:
      a,b,out = alloc(m*k*2),alloc(n*k*2),alloc(m*n*4)
      cmd,task = alloc(4096),alloc(4096,rk.RKNPU_MEM_KERNEL_MAPPING)
      tail = (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS),_cmd(rk.TARGET_VERSION,0),
              _cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd))
      ctypes.memset(cmd.addr,0,cmd.size)
      width = round_up(ceildiv(n,args.cores),32)
      parts = [(offset,min(width,n-offset)) for offset in range(0,n,width)]
      position = 0
      for index,(offset,columns) in enumerate(parts):
        base = {(word>>48,word&65535):(word>>16)&0xffffffff for word in
                geometry(m,columns,k,a.dma,b.dma+offset*k*2,out.dma+offset*m*4)}
        base[(0x201,rk.REG_CNA_DMA_CON2)] &= 0x0fffffff
        base[(0x1001,rk.REG_DPU_OUT_CVT_SCALE)] = 1
        body = tuple(_cmd(row['target'],row['address'],base.get((row['target'],row['address']),row['value'])) for row in template)
        words = array.array('Q',body+tail)
        ctypes.memmove(cmd.addr+position*8,words.buffer_info()[0],len(words)*8)
        descriptor = rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,cmd.dma+position*8)
        ctypes.memmove(task.addr+index*ctypes.sizeof(descriptor),ctypes.addressof(descriptor),ctypes.sizeof(descriptor))
        position += round_up(len(words),2)
      dev.sync(cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      dev.sync(task,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      for repeat in range(6):
        x = rng.normal(0,.2,(m,k)).astype(np.float16)
        weight = rng.normal(0,.2,(n,k)).astype(np.float16)
        copy(a,x.reshape(m,k//8,8).transpose(1,0,2).copy())
        copy(b,weight.reshape(n//16,16,k//32,32).transpose(0,2,1,3).copy())
        dev.sync(out,rk.RKNPU_MEM_SYNC_TO_DEVICE)
        begin = time.perf_counter()
        subcores = [(0,1)] if args.cores==1 else [(0,0),(0,0),(0,1),(1,1),(2,1)]
        rk.DRM_IOCTL_RKNPU_SUBMIT(dev.fd,flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG,
          timeout=6000,task_start=0,task_number=len(parts),task_counter=0,priority=0,task_obj_addr=task.meta.obj_addr,
          regcfg_obj_addr=0,task_base_addr=0,user_data=0,core_mask=(1<<args.cores)-1,fence_fd=-1,
          subcore_task=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*part) for part in subcores)))
        elapsed = time.perf_counter()-begin
        dev.sync(out,rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        actual = np.frombuffer(to_mv(out.addr,m*n*4),np.float32).copy().reshape(n//4,m,4).transpose(1,0,2).reshape(m,n)
        expected = x.astype(np.float32)@weight.astype(np.float32).T
        row = {'shape':[m,k,n],'cores':args.cores,'repeat':repeat,'submit_ms':elapsed*1000,
               'max_abs_error':float(np.max(np.abs(actual-expected)))}
        report['cases'].append(row)
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=3e-4)
        print(json.dumps(row),flush=True)
    report['complete'] = True
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally:
    open(args.report,'w').write(json.dumps(report,indent=2)+'\n')
    if not dev.poisoned: dev.reset()
    for buf in reversed(owned): dev.free(buf)
    dev.finalize()


if __name__=='__main__': main()
