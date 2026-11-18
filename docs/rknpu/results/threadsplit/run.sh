#!/bin/sh
# Run the sweep repeatedly, recording the clock state around each run.
# Nothing is pinned by this script (no sudo on this host); the state is only
# observed and recorded, and every variant is interleaved so ratios are taken
# between adjacent runs rather than across time.
#
# Validity gate, fixed before the runs were taken: a run is VALID for reporting
# iff (a) the dispatch-only variant DISP has median < 5 us, and (b) no timed
# variant of phase 1 or 2 has more than one tick spike.  DISP does no work at
# all, so any inflation of it is external interference by definition; rejecting
# such a run is not a selection on the result of the kernels under test.
#
# usage: ./run.sh [count]
set -e
cd "$(dirname "$0")"
count=${1:-2}

clocks() {
  echo "# clocks $1:"
  for p in 0 4 6; do
    d=/sys/devices/system/cpu/cpufreq/policy$p
    echo "#   policy$p gov=$(cat $d/scaling_governor) min=$(cat $d/scaling_min_freq)" \
         "max=$(cat $d/scaling_max_freq) cur=$(cat $d/scaling_cur_freq)"
  done
  echo "#   ddr cur=$(cat /sys/class/devfreq/dmc/cur_freq) min=$(cat /sys/class/devfreq/dmc/min_freq)"
  echo "#   npu cur=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq)"
  echo "#   temp=$(cat /sys/class/thermal/thermal_zone0/temp)"
  echo "#   load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

i=0
while [ "$i" -lt "$count" ]; do
  clocks "before run $i" >  "run-$i.log"
  ./bench 20 20            >> "run-$i.log" 2>&1
  clocks "after run $i"  >> "run-$i.log"
  i=$((i + 1))
done
echo "wrote $count run logs"
