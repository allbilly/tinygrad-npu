"""Record real per-core RKNPU load, interrupt counts, and clocks around a command.

The driver reports busy-time load in one-second windows. Devfreq's load field is
deliberately excluded: this board's driver sets busy_time=total_time=1 at init.
Use --docker-image only when debugfs is unreadable; it mounts the NPU debugfs
directory read-only and runs the installed converter image's existing Python.
"""
import argparse, json, os, re, statistics, subprocess, threading, time
from pathlib import Path


def parse_load(text:str) -> list[int]:
  values = re.findall(r'Core([0-2]):\s*(\d+)%',text)
  if [int(core) for core,_ in values]!=[0,1,2] or any(not 0<=int(value)<=100 for _,value in values):
    raise ValueError(f'Expected three per-core NPU busy-time percentages: {text!r}')
  return [int(value) for _,value in values]


def snapshot() -> dict:
  lines = [line for line in Path('/proc/interrupts').read_text().splitlines() if 'fdab0000.npu' in line]
  counts = [sum(int(value) for value in line.split(':',1)[1].split() if value.isdecimal()) for line in lines]
  clocks = [*Path('/sys/class/devfreq').glob('*/cur_freq'),
            *Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')]
  return {'time':time.monotonic(),'interrupts':counts,'frequencies':{str(p):int(p.read_text()) for p in clocks}}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--interval',type=float,default=.1)
  parser.add_argument('--duration',type=float,default=5,help='record seconds when no command is supplied')
  parser.add_argument('--docker-image',help='local x86_64 converter image containing /qemu and Python3.10')
  parser.add_argument('command',nargs=argparse.REMAINDER)
  args = parser.parse_args()
  if args.interval<=0 or args.duration<=0: parser.error('Interval and duration must be positive')
  command = args.command[1:] if args.command[:1]==['--'] else args.command
  report = {'complete':False,'command':command,'interval_seconds':args.interval,'driver_load_window_seconds':1,
            'load_source':'debugfs/rknpu/load','load_samples':[],'samples':[]}
  reader = None
  container_name = f'onpu-monitor-{os.getpid()}'
  stop,ready = threading.Event(),threading.Event()
  error:list[str] = []
  def sample_load():
    try:
      if args.docker_image:
        source = ('import json,time\nfrom pathlib import Path\nwhile True:\n'
                  ' print(json.dumps({"time":time.monotonic(),"raw":Path("/npu/load").read_text()}),flush=True)\n'
                  f' time.sleep({args.interval!r})\n')
        nonlocal reader
        reader = subprocess.Popen(['docker','run','--name',container_name,'--rm','--network','none','--cpuset-cpus','0-3','--platform','linux/amd64',
          '--entrypoint','/qemu','-v','/sys/kernel/debug/rknpu:/npu:ro',args.docker_image,
          '/usr/local/bin/python3.10','/usr/local/bin/python3.10','-u','-c',source],stdout=subprocess.PIPE,text=True)
        assert reader.stdout is not None
        for line in reader.stdout:
          row = json.loads(line)
          report['load_samples'].append({'time':row['time'],'core_percent':parse_load(row['raw'])})
          ready.set()
      else:
        while not stop.is_set():
          report['load_samples'].append({'time':time.monotonic(),'core_percent':parse_load(Path('/sys/kernel/debug/rknpu/load').read_text())})
          ready.set()
          stop.wait(args.interval)
    except Exception as exc:
      error.append(str(exc))
      ready.set()
  thread = threading.Thread(target=sample_load,daemon=True)
  thread.start()
  if not ready.wait(30): error.append('Per-core load reader did not become ready')
  child = None
  try:
    if error: raise RuntimeError('; '.join(error))
    report['start'] = time.monotonic()
    if command: child = subprocess.Popen(command)
    while True:
      report['samples'].append(snapshot())
      if child is not None and child.poll() is not None: break
      if child is None and time.monotonic()-report['start']>=args.duration: break
      if error: raise RuntimeError('; '.join(error))
      time.sleep(args.interval)
    report['end'] = time.monotonic()
    report['exit_code'] = child.wait() if child is not None else 0
    report['irq_delta'] = [b-a for a,b in zip(report['samples'][0]['interrupts'],report['samples'][-1]['interrupts'])]
    loads = [row['core_percent'] for row in report['load_samples'] if report['start']<=row['time']<=report['end']]
    report['core_load'] = [{'mean_percent':statistics.mean(row[i] for row in loads),'max_percent':max(row[i] for row in loads)}
                           for i in range(3)] if loads else []
    report['complete'] = report['exit_code']==0 and bool(loads)
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally:
    stop.set()
    if child is not None and child.poll() is None:
      child.terminate()
      child.wait()
    if reader is not None:
      subprocess.run(['docker','stop','--timeout','1',container_name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,check=False)
      reader.wait(timeout=10)
    thread.join(timeout=2)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
  if not report['complete']: raise SystemExit(report['exit_code'] or 1)


if __name__=='__main__': main()
