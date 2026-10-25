"""Reset the exclusively owned NPU before entering a vendor runtime process.

Run under flock on /tmp/rk3588_npu_submit.lock. Independent runtimes must not
inherit another runtime's on-chip cached state or partial register assumptions.
"""
import os, sys
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.system import FileIOInterface


def main():
  if len(sys.argv)<2:raise SystemExit('Need executable and arguments')
  device=FileIOInterface(os.getenv('ROCKCHIP_DRM','/dev/dri/card1'),os.O_RDWR)
  rk.DRM_IOCTL_RKNPU_ACTION(device,flags=rk.RKNPU_ACT_RESET,value=0)
  os.execv(sys.argv[1],sys.argv[1:])


if __name__=='__main__':main()
