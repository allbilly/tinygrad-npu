"""Check domain ownership and cleanup with exclusive access to the RK3588 NPU."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, selectors, subprocess, sys
from pathlib import Path


def child():
  import numpy as np
  from tinygrad import Device, Tensor
  from tinygrad.llm.kernels.rockchip_int8 import raw_int8_matmul
  Device['ROCKCHIP']
  print(json.dumps({'owned':True}),flush=True)
  if sys.stdin.readline().strip()!='run':raise ValueError('Missing ownership handoff signal')
  rng=np.random.default_rng(81026)
  source=rng.integers(-128,128,(1,512),dtype=np.int8)
  values=rng.integers(-128,128,(96,512),dtype=np.int8)
  weight=Tensor(values.reshape(3,32,16,32).transpose(0,2,1,3).copy(),device='ROCKCHIP').realize()
  payload=np.zeros(64+512,dtype=np.uint8)
  payload[:4]=np.array([1],dtype=np.float32).view(np.uint8)
  payload[64:]=source.ravel().view(np.uint8)
  actual=raw_int8_matmul(Tensor(payload,device='CPU'),weight,1,96).numpy()
  np.testing.assert_array_equal(actual,(source.astype(np.int32)@values.astype(np.int32).T).ravel())
  print(json.dumps({'exact':True,'sha256':hashlib.sha256(actual.tobytes()).hexdigest()}),flush=True)


def lock_blocked() -> bool:
  with open('/tmp/rk3588_npu_submit.lock','a') as handle:
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return True
    return False


def mapped(address:int) -> bool:
  return any(int(bounds[0],16)<=address<int(bounds[1],16)
             for line in Path('/proc/self/maps').read_text().splitlines() if (bounds:=line.split()[0].split('-')))


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args=parser.parse_args()
  from tinygrad import Device
  from extra.rockchip.benchmark_inference import source_snapshot
  root=Path(__file__).resolve().parents[2]
  result={'complete':False,'source_sha256':source_snapshot(root),'run_token':os.getenv('ONPU_EVIDENCE_RUN','manual'),
          'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'Live buffers, domain finalize, lock handoff and fresh-process exact INT8 output. No model speed or quality claim.'}
  devices=[]
  replacement=None
  try:
    for name in ('ROCKCHIP','ROCKCHIP:1'):devices.append(Device[name])
    buffers=[dev.allocate(4096) for dev in devices]
    assert devices[0].hw is devices[1].hw and all(mapped(buf.addr) for buf in buffers)
    devices[1].finalize()
    result['second_domain_unmapped']=not mapped(buffers[1].addr) and not devices[1].allocations and devices[1].mapped_bytes==0
    result['first_domain_still_owned']=mapped(buffers[0].addr) and devices[0].hw.users==1 and lock_blocked()
    assert result['second_domain_unmapped'] and result['first_domain_still_owned']
    devices[0].finalize()
    result['all_live_buffers_unmapped']=all(not mapped(buf.addr) and not dev.allocations and dev.mapped_bytes==0
                                          for dev,buf in zip(devices,buffers))
    result['shared_fd_closed']=not hasattr(devices[0].fd,'fd')
    result['ownership_released']=not lock_blocked()
    assert result['all_live_buffers_unmapped'] and result['shared_fd_closed'] and result['ownership_released']
    replacement=subprocess.Popen([sys.executable,'-m','extra.rockchip.probe_domains','--child'],cwd=root,
      env={**os.environ,'DEV':'CPU','DEBUG':'0','OPENBLAS_NUM_THREADS':'1'},text=True,
      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    with selectors.DefaultSelector() as selector:
      selector.register(replacement.stdout,selectors.EVENT_READ)
      if not selector.select(30):raise TimeoutError('Replacement NPU owner did not become ready')
    assert replacement.stdout is not None and replacement.stdin is not None
    owned=json.loads(replacement.stdout.readline())
    assert owned=={'owned':True} and lock_blocked()
    # The raw buffer objects intentionally outlive finalize. Late frees must not
    # touch the released descriptor or disturb the replacement process's domain.
    for _ in range(2):
      for dev,buf in zip(devices,buffers):dev.free(buf)
    result['late_frees_while_new_owner_active']=True
    stdout,stderr=replacement.communicate('run\n',timeout=60)
    result['replacement_exit_code']=replacement.returncode
    result['replacement_stderr']=stderr
    result['replacement_result']=json.loads(stdout.strip())
    assert replacement.returncode==0 and result['replacement_result']['exact'] is True
    result['complete']=True
  except BaseException as error:
    result['error']={'type':type(error).__name__,'message':str(error)}
    raise
  finally:
    errors=[]
    if replacement is not None and replacement.poll() is None:
      replacement.kill()
      replacement.wait(timeout=30)
    for dev in reversed(devices):
      try:dev.finalize()
      except BaseException as error:errors.append(repr(error))
    result['cleanup_errors']=errors
    result['complete']=result['complete'] and not errors
    args.report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
  if sys.argv[1:]==['--child']:child()
  else:main()
