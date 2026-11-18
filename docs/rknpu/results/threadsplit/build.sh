#!/bin/sh
# Exact build command for threadsplit/bench.c.
#
# Project flags (extra/rockchip compiled-kernel flags, as recorded for the
# CPU kernels these variants model):
#   -O2 -fPIC -ffreestanding -fno-math-errno -march=armv8.2-a -mcpu=native
# Dropped from the project set because they break a hosted executable:
#   -nostdlib   (no crt0/__libc_start_main -> will not link a normal program)
# Added for this harness only: -lpthread -lm.
set -e
cd "$(dirname "$0")"
clang -O2 -fPIC -ffreestanding -fno-math-errno -march=armv8.2-a -mcpu=native \
      -Wall -Wextra -Wno-unused-parameter \
      bench.c -o bench -lpthread -lm
echo "built: $(pwd)/bench"
