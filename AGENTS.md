# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Do not do amend commits. Always do a new commit if a force push to origin would be required.
- tinygrad has user space PCI drivers for AMD and NVIDIA GPUs. Do not insert the unneeded kernel modules.

## oNPU RK3588 work (this checkout)

`~/pr/tinygrad` is the only worktree on main. Do not create extra worktrees or
experiment subfolders; new code lives under `extra/rockchip/`.

`~/oNPU` and `~/tinygrad` are **reference only**: read them for evidence and
design, never run their runners or import their trees.

### Running

- Interpreter: `~/pr/tinygrad/.venv/bin/python` (3.12, numpy, tinygrad installed
  editable). Do not use any other venv or system python.
- No `PYTHONPATH` needed: the editable install resolves `tinygrad` to this tree.
- Baseline device: the model runs on the CPU host with every `nn.Linear`
  offloaded to the NPU GEMM (`bench_llm.py --npu`). `DEV=ROCKCHIP` is the NPU
  device itself; `DEV=CPU` alone is a reference only, not the baseline.

### NPU device

- `tinygrad/runtime/ops_rockchip.py` and `tinygrad/runtime/autogen/rockchip.py`
  are the ported RK3588 device. Only the `Compiled`/`Program` signature changes
  were needed: `renderers` is a class attribute and `Program` takes `links`.
- Single owner: it takes `/tmp/rk3588_npu_submit.lock` and opens
  `/dev/dri/card1`. Run one NPU process at a time.
- Smoke test: `~/pr/tinygrad/.venv/bin/python extra/rockchip/smoke_matmul.py`
  must print `max_abs_err=0.0000 rel=0.00000`.
- Matmul layouts: lhs `(m,k) -> (k/8,m,8)`, weight
  `(n,k) -> (n/16,k/32,16,32)`, out `(n/4,m,4) -> (m,n)`.
- Limits (FP16 recipe as ported): `m<=64` (`m<=128` in `row_major`), `k<=4096`,
  `n<=4096` per core (`n<=8192` per core when `m==1`; with `cores=3` the call
  limit is 12288 / 24576), `k%32==n%32==0`. These are recipe bounds, not silicon
  maxima. The ported code also enforces `m*k <= 65536` through
  `banks = ceil(m*k*2/32768) <= 4` (written into `REG_CNA_CBUF_CON0`), so the
  wrapper uses `row_tile = min(64, 65536//k)` and splits K at 4096.
- Reference evidence (`~/oNPU/CHATGLM_DECODE_ANALYSIS.md`): the measured CBUF
  wall is `k <= 11264` (22 groups of 512) for INT8 and `m=64, k=5632` runs, so
  the 4096 cap is conservative by ~2.75x; `m=64, n=4608` needs 3 cores because
  4096 is the per-core prefill width. The FP16 wall was not re-measured here.

### Timing protocol

- 128-token prompt, 64 greedy decode tokens, 1 warm-up + 3 measured.
- Report every trial and the median. The oNPU reference tree saw a 6-15x TTFT
  spread from trial order; in this tree the CPU trials were within 6%. Never
  pick a trial post hoc without a pre-registered stabilization rule.
- `extra/rockchip/bench_llm.py` implements this and writes `report.json`.
- revalidate timing with a 52 to 55 oC cold start, depends on idle room temperature

### No config flags

- Derive config or use a module constant with the measurement in a comment; never hide it behind an env var.

### Commits

- Keep changes as an uncommitted working-tree diff. Do not commit unless asked.
- If commits are requested: small, focused, no amend commits (upstream rule).

### Markdown

- All markdown table shd fix spacing to plan text view readability

### Opencode

- In a interactive session, just plan dont run tasks without approval
- In a goal session, do all tasks with ur suggested actions do not ask user questions except remove non cacahe files need explict user approval 
- In opencode, u can create subagents teammates with roles and offload tasks to them, all doc review task shd do in subagents