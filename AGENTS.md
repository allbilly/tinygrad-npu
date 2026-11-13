# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Do not do amend commits. Always do a new commit if a force push to origin would be required.
- tinygrad has user space PCI drivers for AMD and NVIDIA GPUs. Do not insert the unneeded kernel modules.

## RKLLM benchmark status

- **LFM2.5-VL 450M already wins both phases** (oNPU 70.13 ms prefill / 67.59 tok/s decode vs local RKLLM 103.70 / 54.58). Do not re-convert or re-benchmark LFM; skip LFM cohorts unless its execution path changes.
- **ChatGLM3-6B prefill already wins** (1261.5 vs 1372.5 ms, 1.09x). Do not re-test ChatGLM prefill; only its **decode** is open (4.536 vs 5.013 tok/s).
- Remaining target: **MiniCPM3-4B** both phases (5.342 vs 5.785 tok/s decode; 2412.5 vs 1461.6 ms prefill).
- Fair comparison is W8A8 only: no speculative draft model, no lower precision, no changed algorithm basis.
- CPU tests use `DEV=CPU ROCKCHIP_TEST=0`. Hardware benchmarks are single-owner; do not overlap them with tests or other benchmarks.
- Pending changes live in worktrees, not the main tree. Merge only after CPU goldens pass.
