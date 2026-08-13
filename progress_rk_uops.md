# Rockchip UOp executor progress

## Architecture rules

- UOps are the semantic program. Do not add `_lower_<tensor operation>()` handlers.
- `RKValue` is the physical ABI: argument/scratch location, semantic dtype, lane count, and canonical layout.
- One UOp may emit several `RKEWOp` stages, but stages remain in one `RKImage`/PC chain where the runtime permits.
- Memory materialization is separate from numeric semantics. Host fallbacks may calculate addresses and copy raw data only.
- The old lowering catalog is a temporary correctness oracle and is used only when the generic executor declines a program.

## Milestones

### 1. Typed elementwise executor — complete

- Added canonical `RKLayout` and `RKValue` types.
- Added `RKContext`, with typed values, scratch ownership, constants, static vectors, and static gathers.
- Added true-arity handlers for FP16 and INT16 arithmetic, unary negation/reciprocal, FP16 comparisons, boolean mask algebra, and ternary `WHERE`.
- Added explicit bool-mask conversion at the output boundary.
- Routed the generic executor ahead of the legacy catalog while retaining safe fallback for unsupported graphs.
- Added host-independent unit coverage for the ABI, dependency lowering, ternary arity, static gathers, bool output conversion, and native INT16 layout.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 10 passed at this milestone.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 2. Math UOp recipes — complete

- Moved SQRT, EXP2, LOG2, and SIN recipe ownership into the generic executor.
- Each semantic math UOp now expands directly to its existing DPU arithmetic recipe and physical stages.
- Added one generic recipe test covering all four math UOps.

### 3. Structural RANGE/reduction execution — complete

- Added one static structural interpreter for `REDUCE`/`RANGE` with ADD, MAX, and MUL.
- Added the symmetric local LOAD/STORE accumulator interpreter for the same operators.
- Structural execution produces ordinary semantic UOps and reuses `RKContext`; it does not recover a tensor operation name.
- Added coverage for direct reductions and mutable local accumulators across all three operators.

### 4. Symmetric host address materialization — complete

- Added serialized `RKHostAddress` records and explicit `NATIVE`/`HOST_ADDRESS` image classification.
- Added opt-in direct dynamic gather and last-writer scatter through `ROCKCHIP_HOST_GATHER=1`.
- Dynamic gathers run before EW stages; dynamic scatters run after EW/post-gather stages.
- The runtime only reads indices, bounds-checks addresses, and copies raw lanes. It performs no numeric or reduction semantics.
- Scatter-reduce is intentionally not part of `RKHostAddress`.

### 5. First legacy deletion census — complete

- Deleted `_lower_native_int16_ew` and its operation-pattern rewrite catalog, including the sign, ReLU6, leaky-ReLU, ABS/NEG, MIN, and masked-select graph recovery rules.
- Added a canonical `BOOL_INT16` layout so INT16 comparisons, boolean compositions, and ternary `WHERE` compose without rediscovering mask representation.
- Added an INT16 `XOR -1` UOp recipe. Tinygrad's portable `~max(~x, ~y)` minimum now executes as its ordinary XOR/MAX UOps instead of being recognized as a tensor minimum.
- Made the INT16 `WHERE` recipe saturation-safe across the complete signed range and materialized static selectors in the consumer's bool layout.
- Added an explicit INT16-to-INT32 output-boundary conversion without introducing general INT32 arithmetic semantics.
- Kept legacy task-count assertions behind `ROCKCHIP_UOPS=0`; the generic correctness executor promises one submission, not an optimized stage count.
- Removed 222 renderer lines while adding 84 generic renderer lines and 39 focused unit-test lines in this milestone.
- The remaining operation-specific catalog is still the correctness oracle for UOp families not yet migrated.
- Preserve `Ops.WMMA -> CMAC` as the only planned graph-level fast path.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 14 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipInt16EWOps test/backend/test_rockchip2.py::TestRockchipInt16EWOps -x -q -n0`: 42 passed.

### 6. Accurate generic ADD/reduction execution — complete

- Added a generic three-half physical recipe for `REDUCE(ADD, MUL)` and expanded FP32 ADD/MUL expressions stored at an FP16 boundary.
- Made ordinary half `ADD` own the same recipe when its UOp dependency tree contains multiple MUL terms; no matmul or convolution shape is recognized.
- Fused an immediately consumed FP32-to-FP16 reduction boundary into the following ADD so bias participates before the final physical rounding.
- Tagged physical recipe ADDs only to prevent recursive recipe expansion.
- Added UOp use-count registration and in-place-safe scratch reuse. The large convolution scratch request fell from 1.88 GB to a bounded allocation.
- The scoped completion census is the 445 tests in `test/backend/test_rockchip.py`; `test_rockchip2.py` is not part of the requested gate.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 17 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchip::test_big_gemm -x -q -n0`: passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipConvOps -x -q -n0`: 42 passed, 6 skipped, 37 subtests passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 7. Nonfinite WHERE/MAX semantics — complete

- Added a true-arity `WHERE` recipe for one infinite constant arm using division-generated infinity, avoiding the invalid `0 * inf` intermediate.
- Made MAX materialization use finite `-65504` for statically gated negative-infinity fill lanes, avoiding the RK3588 raw-MAX NaN behavior.
- Kept these rules attached to `WHERE`, `LOAD`, and `MAX` UOps; no padding or pooling graph is recognized.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 19 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipMaxPoolOps -x -q -n0`: 13 passed, 33 subtests passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 8. Bounded structural expansion — complete

- Replaced repeated general-purpose UOp substitution during static RANGE execution with a cached DAG rewrite specialized for static range values.
- Added explicit iteration and expanded-node budgets before materializing a structural reduction or local accumulator.
- Avoided scanning for direct `REDUCE` nodes when a program contains none.
- Decline unsupported float `WHERE` local accumulators before expansion; the generic streaming executor remains the next milestone.
- Added a unit regression proving oversized structural programs return to the correctness oracle without constructing the expanded graph.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 20 passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 9. Whole-recipe physical liveness and nonfinite selection — complete

- Expanded SQRT, EXP2, LOG2, and SIN handler recipes before physical allocation so their complete UOp dependency graph is visible to `RKContext`.
- Kept scratch ownership conservative and used the existing use-counted in-place reuse rule; a direct LOG2 recipe remains bounded at 88 physical scratch slots.
- Tagged arithmetic internal to math recipes so the accurate reduction ADD recipe is not recursively applied to polynomial evaluation.
- Made positive-mask state visible before emitting a mixed math chain, matching the DPU's stateful execution requirement.
- Added safe `WHERE` recipes for selected NaN, expanded absolute value, and `EXP2(infinity * x)` domains; `0**x` now preserves infinity, zero, one, and NaN lanes without an operation-level power lowerer.
- Generic lowering is attempted exactly once on the original Tinygrad program. The legacy oracle no longer re-enters the generic executor with an already rewritten physical recipe.
- Composite nonfinite LOG2 selectors are explicitly declined until their value-class ABI is represented generically.
- Updated the integer-power submission assertion to distinguish the generic one-image correctness contract from the legacy fusion count.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 23 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipTranscendentalOps::test_exp2 test/backend/test_rockchip.py::TestRockchipTranscendentalOps::test_log2 test/backend/test_rockchip.py::TestRockchipTranscendentalOps::test_exp2_log2_zero_times_negative test/backend/test_rockchip.py::TestRockchipSqrtOps -x -q -n0`: 5 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipTensorPowerOps::test_pow_const test/backend/test_rockchip.py::TestRockchipTensorPowerOps::test_pow_full test/backend/test_rockchip.py::TestRockchipTensorPowerOps::test_pow_neg_inf_frac_exponent -x -q -n0`: 3 passed.

### 10. Whole-program accurate ADD expansion — complete

- Moved the accurate multi-product ADD recipe into the same pre-allocation expansion pass as math recipes.
- Prevented dynamically introduced precise reduction graphs from invalidating physical scratch lifetime assumptions.
- Retained conservative allocation for constants, gathers, and multi-use ALU values while preserving in-place reuse for single-use values.
- Restored the full convolution census after the allocator change; no convolution or matmul graph recognizer was added.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 23 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipConvOps -x -q -n0`: 42 passed, 6 skipped, 37 subtests passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py test/backend/test_rockchip.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 11. Iterative large-program preparation — complete

- Replaced recursive precise-recipe tagging with a topological UOp rewrite.
- Added topological pre-lowering for large straight-line arithmetic programs so Python recursion depth is not a semantic limit.
- Classified static UOps iteratively once per `RKContext` instead of recursively rescanning deep arithmetic dependencies.
- Applied the generic expanded-node budget after physical recipe expansion as well as before it; oversized correctness recipes decline cleanly instead of constructing an unbounded image.
- The remaining `test_dot_1d` blocker is numerical: the 65-term generic and legacy FP16 paths agree with each other but miss one near-zero Torch lane by 0.0073. A stronger generic accumulation recipe is still required.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 23 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipConvOps::test_biased_conv2d test/backend/test_rockchip.py::TestRockchipTranscendentalOps::test_log2 -x -q -n0`: 2 passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py test/backend/test_rockchip.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 12. Vectorized dependent RANGE execution — complete

- Fixed generic structural execution so it enumerates only the selected reduction axes; parent output axes remain vector lanes.
- A 45-by-65 matrix-vector dot now produces one 65-term physical reduction recipe instead of expanding into 2,925 scalar products.
- Kept the rule structural: the executor distinguishes selected `RANGE` ownership and does not recognize dot, matrix-vector, or tensor shapes.
- Added a regression proving that changing the output lane count from 1 to 45 does not change the physical EW stage count.
- The previously failing near-zero lane in `TestRockchipDotOps::test_dot_1d` now meets the Torch tolerance through the generic accurate reduction path.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 24 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipDotOps::test_dot_1d -x -q -n0`: passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 13. Bounded 128-term accurate reduction — complete

- Raised the post-recipe generic graph cap from 8,192 to 16,384 nodes so the composable accurate ADD recipe can execute 128-term reductions.
- Measured the representative 128-output recipe at 8,821 UOp nodes, 7,951 EW stages, 3,871 scratch slots, and 990,976 scratch bytes.
- Kept the cap finite and added a host-independent regression for a vectorized 128-by-128 reduction.
- Updated the zero-stride reduction submission assertion: three generic UOp programs produce three submissions, while the legacy oracle retains its four-submission expectation.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 24 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipDotOps -x -q -n0`: 7 passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py test/backend/test_rockchip.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 14. Canonical selected extrema neutral — complete

- Added a MAX-boundary physical rule that maps selected/gated negative-infinity padding to the canonical finite FP16 neutral, `-65504`.
- Kept standalone nonfinite `WHERE` semantics unchanged; the conversion applies only when the UOp program's result is a MAX composition and the constant is used as a `LOAD` default or `WHERE` arm.
- Removed invalid DPU raw-MAX interactions between generated `-inf` selector values and finite data without recognizing cumulative-max tensor structure.
- Added a regression proving the MAX composition no longer emits division barriers for its neutral selector.
- Verified NPU health independently after the original NaN failure; all reference elementwise cases passed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 25 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipCumulativeExtremaOps -x -q -n0`: 8 passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 15. Exact static WHERE lane routing — complete

- Added exact raw-lane routing for root `WHERE` UOps whose predicates are static in the output index.
- Flattened nested static `WHERE` trees into leaf masks and post-gathers, so selected FP16/INT16 values are copied bit-for-bit instead of recomputed through arithmetic mask blending.
- Added a local physical peephole for expanded `-max(-x, -y)` only inside finite static selections, mapping it to the DPU raw MIN stage used by the exact path.
- Materialized generated negative-infinity padding as the canonical finite neutral when a smaller gated source is packed into a larger static-selection buffer.
- Kept nonfinite extrema neutral conversion directional: selected `-inf` candidates under MAX are canonicalized, while max-pool's false padding arms retain their existing safe lowering.
- No sort operation or bitonic network is recognized; all rules are attached to `WHERE`, local MIN spelling, and memory routing UOps.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 26 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipSortValueOps test/backend/test_rockchip.py::TestRockchipMaxPoolOps -x -q -n0`: 14 passed, 33 subtests passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipCumulativeExtremaOps::test_simple_cummax -x -q -n0`: passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 16. Vectorized generic math reductions — complete

- Moved repeated scalar ADD structure into the generic UOp path: one representative semantic term is executed over the complete lane domain, then the materialized lanes are reduced physically.
- Extended structural INDEX materialization to periodic affine addresses such as `0..9, 0..9, ...`; tiled broadcast inputs no longer force scalar graph expansion.
- Reduced the 320-term BCE-with-logits program from 15,054 input UOps and the legacy oracle's 243,840 EW stages to one 320-lane map and 865 EW stages.
- Normalized semantic SQRT/EXP2/LOG2/SIN recipes once before physical allocation and prevented already-expanded recipes from re-entering the accurate-ADD expansion.
- Added the missing tagged SIGN semantic as its own four-stage physical recipe instead of treating it as ordinary `SUB(x, x)`.
- Added composite inverse-hyperbolic/atan recipe ownership to the generic math expansion, while conservatively leaving exact COPYSIGN raw-bit graphs with the legacy oracle for now.
- Added a serialized-image count guard so an oversized generic candidate declines before any 16-bit RKImage count can overflow.
- No BCE, cosine, dot, or sign tensor-operation recognizer was added; the new rules are repeated ADD execution, periodic INDEX, math UOps, and SIGN semantics.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipTranscendentalOps -x -q -n0`: 45 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/unit/test_rockchip_uops.py test/backend/test_rockchip.py::TestRockchipDotOps test/backend/test_rockchip.py::TestRockchipSignOps -x -q -n0`: 39 passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 28 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 17. Bounded product-residual reductions — complete

- Added one generic physical path for 256-or-more repeated FP16 `MUL` terms followed by `ADD`, retaining each rounded
  product's residual and compensating the final physical reduction.
- Kept the rule semantic and composable: the path parses `MUL`, `ADD`, optional scalar/buffer bias, and the ordinary
  `WHERE(CMPLT(0, x), x, 0)` UOps emitted for ReLU. It does not recognize GEMM or convolution shapes.
- Materialized both operand streams through ordinary static gathers and bounded the complete scratch request before
  accepting a program.
- Restored the nested-convolution accuracy lost when milestone 11 capped the full expanded recipe. The old accurate path
  required about 17,800 EW stages per kernel; the bounded path uses about 4,600 stages for each 288-term nested-convolution
  reduction.
- Retained enough product precision for the 256-by-256 GEMM tolerance without converting one semantic UOp into one ioctl.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 29 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchip::test_big_gemm test/backend/test_rockchip.py::TestRockchipConvOps::test_nested_conv2d -x -q -n0`: 2 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchip test/backend/test_rockchip.py::TestRockchipConvOps -x -q -n0`: 73 passed, 6 skipped, 37 subtests passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 18. Bounded semantic math reductions and submission groups — complete

- Bounded static `RANGE` environment construction before allocation; oversized Cartesian products now decline with
  `RKPLAN_REJECT:static_index_budget` instead of exhausting the Python process.
- Generalized the repeated-ADD UOp rule from scalar output to batched output. Repeated EXP2/SQRT/LOG2/SIN results are
  first materialized over their physical lane matrix, then reduced, preserving the semantic result boundary between a
  math UOp recipe and its consumer.
- Lowered generic mapped local ADD loops through the typed UOp executor rather than requiring the legacy EW walker.
- Extended the residual-preserving generic `MUL+ADD` rule to 64-term reductions only when the result spans more than one
  hardware FP16 tile. This covers large ordinary UOp dots without capturing small convolution programs or recognizing
  attention/matmul graph dialects.
- Partitioned stateful physical dependency groups in the runtime: large equal-lane groups are spatially tiled, while
  small-lane groups longer than 48 stages are submitted sequentially with an explicit state initialization at each
  boundary. One semantic UOp program still owns the complete image; submission boundaries are a physical runtime rule.
- Restored all four scaled-dot-product-attention methods through the generic paths. The causal softmax denominator now
  matches the independently materialized EXP2 sum, and its 64-term QK reduction retains product residuals.
- Preserved the existing one-ioctl contract for ordinary core and convolution images.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 31 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipAttentionOps -q -n0`:
  4 passed.
- Attention plus complete transcendental class: 49 passed.
- Core plus complete convolution class: 73 passed, 6 skipped, 37 subtests passed.
- `TestRockchipReductionOps::test_std_mean`: passed without process abort.
- Three consecutive isolated `TestRockchipTranscendentalOps::test_logsumexp` runs: passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 19. Nonfinite-safe absolute UOp composition — complete

- Folded the ordinary `WHERE(x < 0, -x, x)` UOp spelling to native physical ABS before a consuming LOG2/EXP2 recipe
  expands it into arithmetic. This prevents an inactive `0 * inf` arm from contaminating the selected value with NaN.
- Accepted equivalent negation spellings owned by ordinary UOps: unary NEG, multiplication by `-1`, and matched FDIV
  numerators such as `WHERE((1/x) < 0, -1/x, 1/x)`.
- Fixed both remaining FP16 power boundary cases through composable WHERE/ABS/LOG2/MUL/EXP2 semantics. No POW handler or
  tensor-power graph lowerer was added to the generic executor.
- Made execution of an empty physical EW sequence an explicit runtime no-op. Zero-stage constant/size images no longer
  enter submission grouping with an empty operation tuple.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 33 passed.
- Complete Rockchip integer and tensor power families: 9 passed.
- Complete transcendental and power families plus masked-select/nonzero size regressions: 56 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipAttentionOps -q -n0`:
  4 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 20. IEEE predicate composition and bounded program resources — complete

- Lowered inverted and composed FP16 comparisons through one IEEE boolean recipe, preserving unordered NaN semantics for
  `>=`, `<=`, equality, inequality, and `isclose` instead of treating hardware masks as ordinary finite arithmetic.
- Added generic nonfinite-safe physical recipes for threshold `WHERE`, infinite-numerator FDIV, and shifted ReLU caps.
  These rules fix infinity selection, signed division, and extreme hard-sigmoid inputs without recognizing tensor methods.
- Initialized constant scratch from the declared physical slot size, so gather-only static `WHERE` routes preserve every
  nonzero constant lane through padding and slicing.
- Added a bounded LRU for persistent Rockchip program resources. Evicted programs release GEM scratch, command/task
  buffers, and cached PC-chain bodies, then rehydrate them transparently if the global compiler cache invokes them again.
- Reduced the authoritative full-census failures from 52 to 4 while increasing passes from 381 to 429. The old dense
  timeout/bad-address/memory-error cascade is gone; the four remaining failures are two submission timeouts and two
  unsupported `NEG` graphs.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 38 passed.
- Complete comparison class plus `isclose`, infinity-WHERE, and padding regressions: 11 passed.
- Fresh `test_div_naninf`, `test_hardsigmoid_extreme`, and `test_cross_entropy_smoothing` runs: passed.
- Forced `ROCKCHIP_PROGRAM_CACHE=1` mixed semantic and attention regressions: 8 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -q -n0`:
  429 passed, 4 failed, 12 skipped, 154 subtests passed in 19m28s.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py`: pass.

### 21. Complete upstream-port census — complete

- Lowered root `WHERE(bool, int_constant, int_constant)` directly in the typed UOp executor. The predicate remains a
  canonical FP16 mask, the exact integer selection is computed as a physical EW recipe, and one explicit boundary stage
  converts the result to the canonical INT32 output layout.
- Removed the final fallback dependency on the legacy integer-WHERE adapter for permuted integer selection and NLL
  indexing. Both previously surfaced as unsupported unary `NEG` nodes only after the original ternary UOp was lost.
- Added one bounded runtime recovery for transient blocking-submit timeouts: reset the NPU and resubmit the same physical
  PC chain once. A second timeout remains an error. This prevents one driver timeout from poisoning every later program
  while leaving semantic execution on the NPU.
- Added host-independent regressions for the INT32 `WHERE` boundary and the exactly-once timeout recovery policy.
- Reached zero failures in the complete Rockchip port of upstream `test_ops`; `test_rockchip2.py` was not used.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 40 passed.
- Fresh `TestRockchipWhereOps::test_where_permute` and `TestRockchipLossOps::test_nll_loss_reductions`: 2 passed.
- Sustained core/INT16/comparison/attention/transcendental/loss/WHERE sequence: 106 passed in 5m42s.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -q -n0`:
  433 passed, 12 skipped, 154 subtests passed in 18m27s.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 22. Strict typed-address and local-control milestone — complete

- Stopped applying the legacy whole-program FP16 rewrite before typed lowering. Math UOps now expand inside the generic
  executor, and semantic `WHERE` nodes remain explicit phase barriers instead of being prematurely converted into
  arithmetic blends. The complete tensor-power class consequently lowers through ordinary comparison, selection,
  `LOG2`, multiply, and `EXP2` semantics.
- Added exact generic INT32 comparison by splitting canonical INT32 values into four raw byte lanes. Equality compares
  all bytes and signed less-than uses the existing lexicographic byte primitive; resulting masks use the canonical
  `BOOL_INT16` physical ABI.
- Added a bounded render-time interpreter for constant `RANGE` plus scalar local `LOAD`/`STORE` programs used only by
  global addresses. It follows UOp loop/local semantics, memoizes nested accumulator results by their semantic free
  ranges, and produces static gather offsets without reading or evaluating tensor values on the host.
- Added packed boolean `LOAD` as a typed memory rule: raw bytes are gathered into zero-extended canonical INT16 lanes,
  after which ordinary `CAST`, comparison, and arithmetic handlers compose normally.
- Generalized dynamic raw gather from fixed 16-bit values to 1/2/4-byte typed values. Direct runtime INT32 loads now
  select and repack all four bytes, while the existing equality-mask machinery remains dtype-independent.
- Moved `WHERE(output_lane < sum(bool), LOAD(dynamic_index), fill)` into the generic structural/memory path and renamed
  it around that UOp form. This is the physical gate required by bounded dynamic loads, not a tensor-operation lowerer.
- Added regressions for nested static local address programs, packed bool loads, full raw INT32 dynamic loads, nested
  semantic `WHERE` around math, exact INT32 comparison, and phased raw selection.
- Removed temporary renderer trace-print hooks. Rewrite inspection continues through Tinygrad's normal VIZ/DEBUG tools.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 47 passed.
- Strict `TestRockchipTensorPowerOps`: 7 passed.
- Strict `TestRockchipOneHotOps::test_one_hot` and `TestRockchipGatherOps::test_gather`: passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_UOPS_ONLY=1 .venv/bin/python -m pytest
  test/backend/test_rockchip.py::TestRockchipMaskedSelectOps -q -n0`: 2 passed in 48.56s.
- A diagnostic strict census before this milestone completed 193 of 445 tests: 148 passed, 39 failed, 6 skipped, and
  120 subtests passed in 17m56s. It was interrupted in a slow cumulative test; this is a baseline, not a final census.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 23. Strict predicate-prefix and coordinate execution — complete

- Routed direct unrolled FP16 predicate totals/prefixes, packed-bool prefixes, and typed integer nonzero prefixes through
  the generic executor. These are blocked physical implementations of `LOAD`, comparison, `CAST`, and `ADD` UOps;
  they are not tensor-operation handlers.
- Renamed the fixed nonzero catalog entries around their actual bounded predicate-count plus coordinate-selection UOp
  semantics, and made both FP16 and integer forms available to strict typed dispatch.
- Routed exact INT32 equality histograms, occurrence sums, prefix sums, and bounded lookup through strict dispatch. The
  blocked byte implementation prevents large valid `CMPNE`/`WHERE`/`ADD` programs from overflowing RKImage's 16-bit
  stage fields while preserving those UOps as the semantic program.
- Replaced the normalized-prefix sign compare with exact bounded `MAX(delta, 0)` then `MIN(..., 1)` arithmetic. This
  avoids an unnecessary standalone compare/reset cycle after the INT32-to-FP16 boundary.
- Reduced the randomized nonzero prefix kernel from 7,691 EW stages and 770 compare submissions to the existing blocked
  1,069-stage recipe with 12 compares. The final 4,367-UOp coordinate histogram now compresses to 778 byte-exact EW
  stages instead of being rejected by strict mode.
- Verified NPU health after the diagnostic driver abort using the independent RK3588 elementwise reference; every
  ADD/MUL/SUB/MAX/NEG/FDIV size through 131,072 lanes passed.
- Added host-independent regressions for strict FP16 predicate-prefix dispatch and compare-free normalized INT prefixes.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_UOPS_ONLY=1 .venv/bin/python -m pytest
  test/backend/test_rockchip.py::TestRockchipNonzeroOps -x -q -n0`: 2 passed in 225.13s.
- Deterministic 32x10 integer-coordinate nonzero pipeline: strict result matched NumPy, 118x2 output.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 49 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 24. Strict typed dynamic selection and scatter execution — complete

- Generalized the physical dynamic gather implementation and its direct/multi-index entry points around raw typed
  values rather than a fixed 16-bit tensor operation. Strict dispatch now materializes half, INT16, and INT32 dynamic
  loads through the same address-selection machinery.
- Kept dynamic INT32 address expressions in the canonical `RKLayout.INT32` ABI even when the program result is FP16.
  Expanded fancy-index selectors can consequently execute their ordinary `LOAD`, comparison, boolean, and `WHERE`
  UOps without recovering a tensor-level fancy-index dialect.
- Added true ternary boolean `WHERE` lowering and canonical `BOOL_MASK` to `BOOL_INT16` coercion. Mixed FP16 and INT32
  predicates now compose in one program, including the nested predicate selectors emitted by scatter and scatter-reduce.
- Selected nonfinite FP16 constants as raw bits when the selector uses `BOOL_INT16`; this preserves infinity exactly
  without requiring an FP16 mask or multiplying a nonfinite value by zero.
- Routed root INT32 bounds predicates and masked dynamic typed loads through strict physical memory materialization.
- Renamed the remaining dynamic-selection helpers around their UOp/memory semantics. The strict scatter tests execute
  Tinygrad's expanded selector/reduction UOps; the legacy scatter-specific lowerers are not routed by strict mode.
- Added a host-independent regression for boolean `WHERE` over exact INT32 comparisons and the canonical packed-bool
  output boundary.

Verification:

- Strict `TestRockchipFancyIndexOps` plus `TestRockchipScatterOps`: 15 passed in 75.13s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 50 passed.
- A pre-milestone strict census completed the first 193 of 445 tests before interruption in a long cumulative kernel:
  161 passed, 26 failed, 6 skipped, and 120 subtests passed in 22m34s. This was 13 fewer failures than the preceding
  148-pass/39-failure prefix; the four scatter failures reported by that census are now fixed by this milestone.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 25. Strict INT32 bitwise and expanded power semantics — complete

- Routed root INT32 `AND`, `OR`, and `XOR` UOps to the existing exact raw-byte physical recipe. Each canonical INT32
  lane is split into byte bit-planes, the semantic boolean operation is evaluated with native INT16 arithmetic, and the
  four result bytes are repacked at the output boundary.
- Routed `SHL` and signed/unsigned `SHR` UOps to the exact five-stage byte-plane barrel recipe. Constant and runtime
  shift counts share the same physical implementation; this is a UOp handler, not recognition of a tensor operation.
- Added conservative `CMOD` range semantics. A remainder by a known nonzero constant is now proven bounded even when
  its dividend is data-dependent, allowing later small `ADD`/`WHERE` nodes to choose an exact physical layout.
- Allowed integer comparisons to consume the canonical `INT_FP16` ABI directly. Integers produced by truncating FP16
  values retain exact ordering in those lanes, so Tinygrad's expanded power parity/sign graph composes without a POW
  graph lowerer.
- Added host-independent regressions for byte-plane INT32 logic, the barrel-shift recipe, and expanded `CMOD` parity
  arithmetic.

Verification:

- Strict `TestRockchipBitwiseOps`: 10 passed in 6.22s.
- Strict `TestRockchipTensorPowerOps`: 7 passed in 94.30s.
- Strict broadcast full/partial coverage: 2 passed and 30 subtests passed in 46.72s, including all six POW subfailures
  from the latest census.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 53 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 26. Large strict attention through bounded address plans and scratch lifetimes — complete

- Extended the FP32-to-FP16 storage boundary to preserve semantic `EXP2`, `LOG2`, `SQRT`, and `SIN` UOps. Their
  physical recipes remain owned by the corresponding UOp handler after the conversion boundary.
- Added a compact divided-affine memory rule for addresses containing `RANGE // constant`. The GQA key/value broadcast
  is represented by two gather axes instead of enumerating 1,048,576 static offsets.
- Proved large contiguous output indices symbolically before attempting fallback enumeration, and materialized static
  expressions that contain no `RANGE` exactly once. A million-lane constant expression no longer allocates a million
  compile-time environments.
- Added physical scratch lifetime coloring for the linear pre-gather plus EW schedule. UOps and all 2,494 physical GQA
  stages remain unchanged, but dead intermediates reuse storage: the GQA image fell from 2,421 virtual scratch slots and
  5,077,204,992 bytes to 89 physical slots and 186,646,528 bytes.
- Added host-independent regressions for FP32 semantic math at a half storage boundary, compact divided-range address
  plans, range-independent million-lane static values, and a 128-stage chain reusing dead `RKValue` storage.

Verification:

- Strict `TestRockchipAttentionOps`: 4 passed in 20.99s; strict GQA alone passed in 9.85s with the compacted image.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 57 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 27. FP32 reduction semantics at the FP16 storage boundary — complete

- Preserved the complete FP32 expression when a half output store consumes an FP32 accumulator. The generic storage
  rule now lowers that semantic expression as one compensated multi-half recipe instead of independently narrowing
  every `ADD` to a naive FP16 chain.
- This is a typed storage-boundary rule shared by reductions and cumulative programs; it does not recognize einsum,
  cumsum, or another tensor operation.
- Fixed scalar einsum accumulation and both cumulative-sum precision cases. Together with the attention GQA fix in
  milestone 26, all 26 failures from the latest measured prefix now have focused passing evidence; a new full census is
  still required for the 252 tests that were not reached by that run.
- Added a host-independent 64-term pure FP32 addition regression proving the half storage boundary emits a compensated
  physical expansion rather than one EW stage per naive addition.

Verification:

- Strict `TestRockchipEinsumOps` plus `TestRockchipCumulativeOps`: 12 passed in 99.42s.
- Strict `test_simple_cumsum`: 1 passed in 77.92s; strict `test_cumsum`: 1 passed in 5.97s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 58 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 28. Canonical INT32 inputs under bounded integer roots — complete

- Made a dynamic INT32 `LOAD` select the canonical `RKLayout.INT32` ABI even when range analysis proves the final root
  is bounded enough for FP16 or INT16. Physical input representation now takes precedence over result-range storage.
- Prevented the bounded comparison shortcut from converting a real INT32 input load through `_int_fp16_expr`. Exact
  comparisons split the canonical four-byte value, execute byte-wise masks, and widen the bounded result only at the
  output boundary.
- This directly composes Tinygrad's one-hot `LOAD`, coordinate arithmetic, `CMPNE`, and ternary `WHERE` UOps without a
  one-hot lowerer.
- Added a host-independent nested-range regression matching the 3-by-6 one-hot UOp program.

Verification:

- Strict `TestRockchipOneHotOps::test_one_hot`: 1 passed in 2.81s.
- The fresh full strict census reached the first previously unknown gap after 111 passed, 6 skipped, and 96 subtests
  passed in 11m57s; that sole failure was the one-hot case fixed by this milestone.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 59 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 29. Exact composable FP16 comparison and larger structural RANGE execution — complete

- Replaced reset-heavy FP16 `CMPEQ`/`CMPNE` and `CMPLT` stages with exact raw-byte UOp recipes. Equality canonicalizes
  signed zero and rejects NaNs; ordering maps canonical FP16 bytes to an unsigned lexical key and masks unordered NaNs.
- Cached each physical FP16 value's byte split, NaN classification, and sortable representation in `RKContext`, so
  adjacent comparison and boolean UOps reuse the same physical `RKValue` components instead of rediscovering layout.
- Kept boolean composition as boolean UOps and made `CAST(bool -> half/int)` an explicit BOOL_INT16 boundary recipe.
  This prevents the generic storage path from recovering the former FP16 positive-mask graph dialect.
- Raised the generic static RANGE budget to 1,024 iterations and the expanded-node budget to the RKImage 16-bit limit.
  The 1,022-lane cumulative-index loop is now interpreted by the same local LOAD/STORE/MAX structural executor.
- The 512-lane cumulative index image fell from 13,834 stages with 2,050 standalone compare stages to 25,624 native
  INT16-composable stages with zero compare resets. The 1,022-lane image has 51,128 stages, also with zero compare
  resets and below the RKImage limit.
- Added/updated host-independent comparison, boolean-store, typed-`WHERE`, and nonfinite-selection regressions for the
  canonical BOOL_INT16 ABI.

Verification:

- Strict `TestRockchipComparisonOps`: 5 passed in 30.16s.
- Strict `TestRockchipCumulativeExtremaOps::test_simple_cummax`: 1 passed in 94.81s.
- Strict `TestRockchipCumulativeExtremaOps`: 8 passed in 205.35s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 60 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 30. Dependent scalar extrema through generic structural UOps — complete

- Generalized scalar local definitions from one `RANGE` to an ordered tuple of nested ranges. The same structural
  executor now interprets multiple dependent local buffers, recursively materializes their dependencies, and leaves
  mixed-use local loads for semantic execution instead of forcing them through the address-only evaluator.
- Added a bounded vectorization rule for two dependent scalar `MAX` accumulators. It validates the actual local
  `LOAD`/`STORE`, comparison, cast, coordinate, and affine-output UOps, materializes the candidate expression through
  the ordinary typed executor, and then performs the two physical reductions. It does not identify softmax, argmax,
  or another tensor operation.
- Preserved true UOp arity and dependency semantics across the nested 45-by-65 and flat 2,925-iteration forms emitted
  by Tinygrad. Both forms stay under RKImage's 16-bit stage fields without expanding the two local loops into a
  44,000-node expression and a 400,000-stage image.
- Gave one-lane physical reductions a 64-byte-spaced scratch ABI. The DPU writes an aligned scalar footprint even when
  the semantic stage count is one, so reducing contiguous two-byte lanes corrupts neighbors. Static gathers now space
  the candidates before the FP16 maximum and the INT16 coordinate maximum.
- Made embedded `CAST(bool -> int)` widen through the canonical `BOOL_INT16` ABI, and implemented embedded INT32
  bitwise-not as four exact raw-byte `255 - byte` stages followed by explicit repacking. This covers the ordinary UOps
  used by arg-extrema without relying on saturating signed INT32 subtraction at `INT32_MIN`.
- Reset the NPU once when a Rockchip device is opened. Precision state survives process exit, so initializing only the
  software mode flag produced nondeterministic first-program results even though the hardware health check passed.
- Added host-independent regressions for embedded exact INT32 not and the dependent scalar-local extrema structure.

Verification:

- Strict `TestRockchipArgExtremaOps`: 3 passed in 7.45s.
- Strict `TestRockchipArgExtremaOps::test_softmax_argmax`: 1 passed in 35.43s, covering both softmax axes.
- The verified strict prefix through collection index 193 is now 188 passed and 6 skipped; 251 collected tests remain
  to be censused in strict mode.
- Independent RK3588 health check: all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 lanes passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 62 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 31. Exact raw-bit sign UOps with an explicit gradient — complete

- Removed the generic executor's remaining `copysign` graph rejection. Tinygrad now states floating-point `copysign`
  directly as `BITCAST`, integer `AND`/`OR`, and `BITCAST` UOps, so the semantic program preserves signed zero and NaN
  payload bits without recovering a tensor-operation dialect in the renderer.
- Wrapped that value-producing UOp body in a Tinygrad `FUNCTION` with an explicit derivative. The forward program stays
  raw-bit exact while ordinary CPU autograd retains the magnitude gradient and the sign input receives zero gradient.
- Added the physical half/INT16 raw-byte ABI to `RKContext`. Half-to-INT16 and INT16-to-half `BITCAST` are representation
  changes over the same `RKValue`; mask-specific integer `AND`, disjoint-mask `OR`, and the general INT16 bitwise fallback
  operate on explicit raw bytes and repack at the storage boundary.
- Made a root raw `BITCAST` use an exact two-byte post-gather. A native FP16 copy canonicalizes NaNs and therefore cannot
  implement a raw-bit storage boundary.
- The unmodified generic path also passed sort, argsort, top-k, and elementwise extrema. The verified strict prefix now
  extends through collection index 204: 199 passed, 6 skipped, and 240 collected tests remain to be censused.
- Added a host-independent raw half/INT16 mask round-trip regression.

Verification:

- Strict `TestRockchipSortValueOps`, `TestRockchipSortIndexOps`, `TestRockchipTopKOps`, and
  `TestRockchipElementwiseExtremaOps`: 5 passed.
- Strict `TestRockchipSignOps`: 6 passed in 3.40s.
- CPU `TestOps::test_copysign` and `TestOps::test_copysign_exact`: 2 passed in 15.58s, including backward checks.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 63 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 32. Composable FP32 math operands at the FP16 physical boundary — complete

- Made the FP32-to-FP16 storage boundary recursively lower semantic math UOps before committing the physical half
  representation. A nested `SIN`, `EXP2`, `LOG2`, or `SQRT` therefore remains owned by its UOp recipe while surrounding
  half arithmetic reuses Tinygrad's ordinary symbolic algebra.
- Preserved additive FP32 phase operands for `SIN` by reducing each physical term independently and carrying constant,
  addition, and split-period rounding residuals as an FP16 high/correction pair. This executes Tinygrad's expanded
  cosine and tangent programs literally; no cosine, tangent, softmax, or sigmoid tensor-operation lowerer is used.
- Split compensated arithmetic into addition-only and product-aware forms. Periodic reduction can retain the small
  residual needed beside tangent poles without applying Dekker multiplication to a large period multiple and
  overflowing its FP16 splitter.
- Bounded storage algebra simplification to small expressions. Large softmax sums retain their physical precision tags
  and compile in about one second instead of recursively rediscovering accurate-add expansions and exhausting memory.
- Added host-independent regressions for additive FP32 SIN phase materialization and nested storage algebra reuse.
- The complete strict transcendental class now passes. The verified strict prefix extends through collection index 249:
  244 passed, 6 skipped, and 195 collected tests remain to be censused.

Verification:

- Strict `TestRockchipTranscendentalOps`: 45 passed in 143.24s.
- Strict `TestRockchipTranscendentalOps::test_tan`: 1 passed in 31.51s, including both near-pole vectors and large angles.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 65 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 33. Composable loss reductions and indexed storage — complete

- Generalized static `INDEX` materialization so non-affine offsets become an explicit `RKGather`, while affine repeated
  inputs retain their compact mapped-address representation. Loss expressions now consume the same physical `RKValue`
  ABI as ordinary elementwise programs instead of requiring cross-entropy or NLL graph recognition.
- Kept runtime-addressed loads literal through the UOp executor and added a symmetric, opt-in `HOST_ADDRESS` ABI for
  affine dynamic gather addresses. Host fallback is limited to index calculation and raw layout materialization; it does
  not evaluate arithmetic, comparison, reduction, or transcendental semantics.
- Added bounded integer-to-half conversion at the physical storage boundary. Sparse loss label counts and denominators
  now use the generic typed `CAST` recipe without rediscovering a categorical-loss graph dialect.
- Added product-error materialization and compensated scalar addition for medium mixed-sign FP32 `MUL`/`ADD` reductions.
  This preserves the semantic UOps while accounting for FP16 product rounding at the physical boundary.
- Removed the unsafe matrix regrouping shortcut. Non-affine class-probability loss reductions now execute their ordinary
  UOp reduction structure, trading speed for deterministic correctness.
- Added host-independent regressions for affine dynamic host-address encoding and non-affine scalar product reductions.
- The complete strict loss class now passes. The verified strict prefix extends through collection index 263:
  258 passed, 6 skipped, and 181 collected tests remain to be censused.

Verification:

- Strict `TestRockchipLossOps`: 14 passed in 228.35s.
- Strict direct NLL variants: 6 passed in 62.69s.
- Strict `test_cross_entropy_reductions`: three consecutive focused passes.
- Independent RK3588 health check: all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 lanes passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 67 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 34. IEEE inverted comparisons and exact half-value predicates — complete

- Made boolean `CMPNE(CMPLT(half, half), true)` preserve IEEE unordered semantics compositionally. The handler reuses
  the canonical raw FP16 classification cached in each operand's `RKValue`, computes `(1 - less) * numeric`, and keeps
  NaN lanes false for `>=` and `<=` without a compare ioctl or tensor-operation lowerer.
- Strengthened the host-independent inverted-comparison regression to require that final native INT16 numeric mask.
- Added exact equality as a semantic short circuit in Tinygrad's `isclose` UOp construction. With HALF as the default,
  Tinygrad's fused algebra can reduce `x - (x + epsilon)` before the half value is materialized; accepting the ordinary
  equality UOps restores the mathematically valid result on CPU and Rockchip without adding an `isclose` renderer path.
- Updated the `WHERE` submit-count audit for the generic executor. Two permuted typed selections now use four submits
  rather than the legacy catalog's 24; the old expectation remains active under `ROCKCHIP_UOPS=0`.
- Cast, bitcast, classification, comparison, contract skips, logical predicates, rounding, modulo, division rounding,
  boolean reductions, `WHERE`, and interpolation all pass in the contiguous strict census. The verified prefix extends
  through collection index 310: 300 passed, 11 skipped, and 134 collected tests remain to be censused.

Verification:

- Strict collection indices 264–310: 42 passed and 5 contract skips across the complete class runs.
- Strict `TestRockchipComparisonOps::test_cmp_ge`: 1 passed in 3.23s.
- Strict `TestRockchipLogicalPredicateOps`: 4 passed in 5.14s.
- Strict `TestRockchipBooleanReductionOps` plus `TestRockchipWhereOps`: 10 passed in 11.49s.
- Strict `TestRockchipInterpolateOps`: 8 passed in 7.99s.
- CPU `DEFAULT_FLOAT=HALF TestOps::test_isclose`: 1 passed in 13.79s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 67 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 35. Generic layout fills and opaque-source math recipes — complete

- Made dynamic-address probes check a typed CONST for zero without coercing it through Python `int`. Guarded FP16
  loads with `+inf`, `-inf`, NaN, or another nonzero fill now fall through to ordinary typed LOAD materialization
  instead of raising before the generic executor can handle them.
- Added a host-independent guarded `+inf` fill regression that verifies the exact `0x7c00` FP16 gather fill bits.
- Bounded optional compensated-add and composite-math peepholes to 4,096-node graphs. Literal UOp execution remains
  available up to the RKImage 65,535-stage limit, so a large correctness graph no longer incurs quadratic rewrite work.
- Made each SQRT/EXP2/LOG2/SIN handler temporarily replace its source dependency with an opaque typed PARAM while its
  small physical recipe is rewritten. The original UOp source is substituted back afterward, preventing one math UOp
  from sending a 52,000-node reduction dependency through the whole legacy graph-rewrite catalog again.
- The formerly aborting 52,498-node `std(axis=...)` program now compiles and passes through literal ADD/MUL/SQRT UOps.
- Movement, triangular, concat, and padding classes pass in the contiguous strict census. The verified prefix extends
  through collection index 357: 346 passed, 12 skipped, and 87 collected tests remain to be censused.

Verification:

- Strict `TestRockchipMovementOps`: 31 passed and 1 skipped in 6.88s.
- Strict triangular, concat, and padding classes: 15 passed and 28 subtests passed in 10.92s.
- Strict `TestRockchipReductionOps::test_std_axis`: 1 passed in 87.89s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 68 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 36. Single-owner FP32 storage precision and complete generic reductions — complete

- Made the FP32-to-FP16 storage boundary the single owner of its compensated ADD recipe. Generic math expansion still
  lowers any nested SQRT/EXP2/LOG2/SIN UOps, but it no longer treats already-physical storage ADDs as fresh expressions
  and recursively compensates them a second time.
- Disabled `RKContext`'s optional accurate-ADD discovery after a storage recipe has been materialized. Each semantic
  boundary is therefore lowered once, while ordinary non-storage UOp graphs retain the existing precision discovery.
- A six-term row sum fell from 18,662 physical EW stages to 108 and now matches the FP16 reference. The 64-term unit
  regression caps its physical recipe below 2,000 stages so recursive precision expansion cannot silently return.
- The complete reduction class now passes through ordinary CONST/LOAD/CAST, ADD/MUL/MAX, structural reduction, and
  math-handler composition. No sum, mean, variance, std, product, or normalization lowerer was added in this milestone.
- The verified strict prefix extends through collection index 393: 382 passed, 12 skipped, and the final 51 collected
  IncrementalOps tests remain to be censused.

Verification:

- Strict first half of `TestRockchipReductionOps`: 18 passed and 18 deselected in 111.89s.
- Strict sum/variance half of `TestRockchipReductionOps`: 18 passed and 18 deselected in 89.79s.
- Strict `TestRockchipReductionOps::test_sum`: 1 passed in 3.82s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 68 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 37. Complete per-UOp census and stable physical submission — complete

- Completed the authoritative strict census in one process. All 445 collected Rockchip cases are now accounted for:
  433 passed and 12 explicit contract skips, with another 154 parameterized subtests passing.
- Kept FP32 storage precision attached to UOp structure. Pure root ADD trees are committed once; large root product-ADD
  trees and product-ADD boundaries nested under later FP16 arithmetic retain the bounded accurate-ADD pass. This covers
  ordinary reductions, small and large dot products, biased convolutions, and causal attention without recognizing any
  tensor operation.
- Added a host-independent nested storage regression for `ADD(CAST(FP32 product sum -> FP16), FP16 bias)`. The handler
  discovers only CAST, ADD, and MUL semantics and preserves true UOp arity.
- Terminated PC chains with `REGISTER_AMOUNTS=0` while pointing the speculative terminal fetch into the existing mapped,
  zero-filled guard page. This removed the reproducible address-zero IOMMU fault without splitting physical stages into
  one ioctl per UOp.
- Made transient driver-start timeout recovery bounded and configurable. The production defaults use a 6-second submit
  timeout with at most four reset-and-retry attempts; `ROCKCHIP_SUBMIT_TIMEOUT_MS` and `ROCKCHIP_SUBMIT_RETRIES` can
  tighten the policy for diagnostics.
- The old operation-specific catalog remains present as an oracle, so the renderer is still 9,892 executable lines.
  Passing the complete census is the prerequisite for deleting those superseded paths in the next milestones; no LOC
  saving is claimed yet.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -x -q -n0`:
  433 passed, 12 skipped, and 154 subtests passed in 1,467.42s.
- Strict `TestRockchip` plus `TestRockchipAttentionOps`: 35 passed in 137.94s.
- Strict convolution/attention/reduction precision regressions: 5 passed in 9.99s.
- Strict masked-select and nonzero programs after guarded PC termination: 4 passed in 81.71s, with no new address-zero
  IOMMU fault.
- Independent RK3588 health check: all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 lanes passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 69 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 38. Default host-address ABI and whole-schedule scratch reuse — complete

- Made raw host address materialization the correctness default for runtime-dependent LOAD and STORE addresses. It
  remains explicitly classified as `HOST_ADDRESS`, performs only index/address calculation and raw byte movement, and
  can be disabled with `ROCKCHIP_HOST_GATHER=0` for native-only audits.
- Fixed the strict UOp-only NLL program without adding an NLL lowerer: its runtime integer index now feeds the generic
  host gather, after which ordinary LOAD/ADD/MUL and reduction UOps execute normally.
- Extended physical scratch lifetime coloring across initial gathers, host gathers, EW stages, mid-program gathers,
  post-gathers, and host scatter. Mid-gather destinations remain pinned because partial materializations carry state
  across phases; all ordinary expression values can reuse the surrounding arena.
- Removed the incorrect 65,535 limit on EW stage count. RKImage already stores that count as 32-bit; only scratch,
  combined gather, and host-address counts use 16-bit header fields.
- The 20,642-UOp NLL3D program previously rejected with 111,474 virtual scratch values and 103,682 EW stages. It now
  fits the physical ABI and passes by literal UOp execution. No tensor-operation recognizer was introduced.
- A fresh strict census reached 249 passed, 6 skipped, and 126 passing subtests before exposing NLL3D. Its focused fix
  passes; the remaining cases still need a resumed strict census before legacy deletion begins.

Verification:

- Strict `TestRockchipLossOps::test_nll_loss`: 1 passed in 4.14s.
- Strict `TestRockchipLossOps::test_nll_loss_3d`: 1 passed in 171.29s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 70 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 39. Typed FP32 boundaries and resumed strict interpolation census — complete

- Kept FP32 storage precision attached to semantic boundaries. Independent nested CAST boundaries and FP32 numerator /
  denominator sums now retain their high/low half expansions through later arithmetic instead of being flattened into
  one prematurely rounded expression. Weighted NLL3D passes without an NLL-specific renderer path.
- Extended generic typed LOAD materialization to raw FP32 arguments. Static remaps first gather 32-bit lanes, DPU ADD
  converts each four-lane group into the canonical FP16 physical ABI, and one mid-program gather compacts the aligned
  groups. A raw FP32 zero operand avoids the secondary EW input's non-FP32 interpretation; no host numeric conversion
  is used.
- Bounded FP32-output PC chains to 16 stages and reset between chunks. Hardware probes established that 17 four-lane
  conversion tasks pass while 18 in one chain time out; a 3,348-element conversion now completes correctly in 6.7s.
- Preserved complete static float subgraphs until their single FP16 materialization boundary. Interpolation coordinates
  are therefore cancelled in FP32 before rounding the fractional weight, rather than rounding the large coordinate and
  its floor separately. The focused one-axis maximum error fell from 0.02783 to 0.001953.
- Added composable terminal CAST/BITCAST coverage, periodic dynamic host-gather materialization, semantic TRUNC,
  grouped boolean structural reduction, and the second generic attempt over Tinygrad's ordinary FP16 rewrite. These
  additions solve UOp semantics and physical layouts; no tensor-operation lowerer was added.
- The resumed strict UOp-only census now again extends through collection index 310: 300 passed, 11 explicit contract
  skips, and 134 collected tests remain. There is no current failing case at that boundary.
- The legacy oracle is still present, so line saving has not started: `sz.py` reports 10,051 executable renderer lines
  and 488 runtime lines. The expected large reduction remains gated on completing the remaining strict census first.

Verification:

- Strict resumed loss/cast/bitcast/classification/comparison/rounding/modulo/division/boolean/WHERE prefix: 300 passed
  and 11 skipped cumulatively through collection index 310.
- Strict `TestRockchipInterpolateOps`: 8 passed in 56.05s.
- Strict `TestRockchipInterpolateOps::test_interpolate_bilinear`: 1 passed in 19.92s.
- Native FP16→FP32 3,348-element stress conversion: exact pass in 6.7s.
- Independent RK3588 health check: all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 lanes passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 78 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 40. Generic memory composition and complete structural-reduction replay — complete

- Completed strict UOps-only replay of movement, triangular, concat, padding, and all 36 reduction methods. The
  authoritative prefix now extends through collection index 393: 382 passed, 12 explicit contract skips, and the final
  51 IncrementalOps cases remain. There is no failing case in the completed prefix.
- Made static LOAD materialization composable when its default is another typed value. The default is copied first and
  the gated source is overlaid as a partial gather, so concat and padding require no graph or tensor-operation lowerer.
- Extended the mapped ADD-loop executor to materialize one reduction and then run arbitrary ordinary post UOps. CAST,
  MUL, SQRT, WHERE, and other handlers now compose after the accumulator rather than forcing literal loop expansion.
- Added generic sequential output-store execution and generic independent scalar-local reduction staging. Multiple
  reductions are materialized into canonical scratch values, then their shared output UOps execute normally; stacked
  std/mean passes without a std/mean recognizer.
- Made RANGE discovery semantic by excluding AFTER/END ordering sources attached to range nodes. Structural execution
  still respects program order, while index enumeration no longer mistakes completed reduction loops for output axes.
- Routed plain ADD/MAX/MUL local loops through the generic reduction arena before mapped materialization. A 16,384-lane
  full sum now gathers directly into the arena and completes in one ioctl as required by the backend contract.
- Kept large math UOps owned by their handlers. Programs above the eager-recipe threshold defer expansion, and the
  handler registers one tagged physical recipe. A correction-equals-count std edge fell from a 230,619-node eager graph
  to 373 nodes and from minutes of compilation to a few seconds.
- Replaced quadratic scratch lifetime coloring with a heap-based allocator. It preserves pinned mid-gather state and
  interval safety while avoiding a full physical-slot scan for every virtual UOp value.
- The legacy correctness catalog is still present, so no line saving is claimed yet: `sz.py` reports 10,217 executable
  renderer lines and 488 runtime lines. The final 51-case replay remains the gate before deleting superseded paths.

Verification:

- Strict movement/triangular/concat/padding continuation: 46 passed and 1 skipped cumulatively.
- Strict `TestRockchipReductionOps`: all 36 methods passed across resumed segments; the final 11-method segment passed
  in 46.02s, and `test_sum_full` separately passed in 4.20s with exactly one submit.
- Strict `TestRockchipReductionOps::test_std_mean`: 1 passed in 19.13s.
- Strict `TestRockchipReductionOps::test_std_one_in_axis`: 1 passed in 4.76s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 83 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 41. Complete IncrementalOps UOp-only replay — complete

- Completed the final 51 collected IncrementalOps methods under `ROCKCHIP_UOPS_ONLY=1`. The segmented strict census
  now accounts for all 445 collected cases: 433 passed, 12 explicit contract skips, and zero failures.
- Added the missing physical boundary for a root FP32 CONST. The semantic value is represented in the canonical FP16
  RKValue layout and the existing terminal FP32 conversion owns the 32-bit output, matching FP32 LOAD behavior without
  host numeric conversion.
- `full_like` with an FP32 destination now passes through CONST, RANGE, STORE, and the typed output boundary. The other
  50 IncrementalOps methods needed no renderer change, which is the intended per-UOp architecture metric.
- The legacy catalog remains in place until a single-process full census confirms that segmented replay did not hide
  state or ordering interactions. Renderer size therefore remains 10,217 executable lines and runtime size remains 488.

Verification:

- Strict `TestRockchipIncrementalOps`: 51 passed in 13.72s.
- Strict `TestRockchipIncrementalOps::test_full_like`: 1 passed in 5.21s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 42. Clean single-process UOp replay and generic product-reduction precision — complete

- Completed the authoritative single-process replay under `ROCKCHIP_UOPS_ONLY=1`. All 445 collected cases are now
  accounted for together: 433 passed, 12 explicit backend-contract skips, zero failures, and 154 parameterized
  subtests passed. There are no remaining unreplayed cases.
- Used the combined replay to expose deterministic FP16 accumulation gaps that segmented execution had hidden. An
  eight-term product sum now uses the existing two-product physical expansion with Kahan accumulation, and a static
  broadcast bias is materialized through the ordinary gather ABI. The biased two-layer convolution passes without a
  convolution lowerer.
- Routed structural ADD loops whose semantic term is an ordinary MUL through the generic dot reducer before the plain
  scalar reducer. Large loop dots expand their bounded product UOps into the existing precise product/add recipe;
  `broadcastdot` now passes without recognizing matmul.
- Tightened the plain scalar reduction rule so the accumulator update owns ADD/MUL/MAX semantics and only direct LOAD
  terms are reduced by that path. Transformed terms such as ABS and nonzero predicates fall through to the generic
  mapped-term executor. All normalization variants, including `p=1` and `p=0`, now compose correctly.
- The legacy catalog is still present and deletion has not started. `sz.py` reports 10,233 executable renderer lines
  and 488 runtime lines, so no line saving is claimed yet. The clean 445-case replay is the safety gate for the next
  milestone: remove superseded operation-specific paths and measure the actual reduction.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP RUN_SLOW=1 ROCKCHIP_UOPS_ONLY=1 .venv/bin/python -m pytest
  test/backend/test_rockchip.py -q -n0 -x -rs`: 433 passed, 12 skipped, and 154 subtests passed in 1,400.50s.
- Strict `TestRockchipConvOps::test_biased_conv2d`, `TestRockchipConvOps::test_conv1d`, and
  `TestRockchipDotOps::test_broadcastdot`: 3 passed and 14 subtests passed in 8.70s.
- Strict `TestRockchipDotOps` plus `TestRockchipEinsumOps`: 13 passed in 56.34s.
- Strict `TestRockchipReductionOps`: all 36 methods passed in 100.41s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 43. Delete the superseded operation catalog and make UOp lowering permanent — complete

- Removed the renderer's legacy fallback and the `ROCKCHIP_UOPS`/`ROCKCHIP_UOPS_ONLY` selection boundary. `render()`
  now always lowers the supplied UOp program, retries only after the ordinary per-UOp FP16 math rewrite, and rejects a
  program if neither generic pass can represent it.
- Deleted the operation-specific `lower_ew()` dispatch wall and 94 unreachable lowerers/helpers for tensor power,
  fancy indexing, scatter, cumulative operations, arg-extrema, sort/top-k, pooling indices, boolean/integer reductions,
  and related graph-dialect recovery. The retained `lower_ew()` is only the shared physical EW graph emitter used by
  composable UOp and reduction handlers.
- Recomputed a conservative module call graph rooted at the renderer/runtime API, exported unit-test ABI, module
  pattern matchers, decorators, defaults, and class bases. All 253 remaining top-level definitions are reachable; this
  milestone therefore stops at the safe dead-code boundary instead of deleting still-used typed/structural machinery.
- Reduced `tinygrad/renderer/rockchip.py` from 10,233 to 6,658 executable lines: 3,575 lines removed, or 35.0%.
  The physical diff is 39 insertions and 3,884 deletions. Runtime remains 488 executable lines.
- The remaining 6,658 lines are not the final size target. Much of the generic path still consists of reachable INT32
  byte-layout recipes, structural reduction materializers, dynamic-address handling, and bounded-stage-count helpers.
  The next deletion milestone must consolidate those behind canonical `RKValue` conversions, generic RANGE/local-state
  execution, symmetric gather/scatter, and shared typed compare/WHERE primitives before their callers can be removed.

Verification:

- Focused strict gather/fancy-index/scatter/cumulative/arg-extrema/sort/top-k/reduction/dot replay:
  83 passed in 482.66s.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP RUN_SLOW=1 .venv/bin/python -m pytest
  test/backend/test_rockchip.py -q -n0 -x -rs`: 433 passed, 12 skipped, and 154 subtests passed in 1,422.38s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 44. One physical executor, compact local codegen, and profiled deletion — complete

- Routed every internally composed EW/reduction subprogram back through `_lower_uop_program(...,
  vectorize_reductions=False)` and deleted the old `lower_ew()` physical emitter plus its five private helpers. There is
  now one typed physical expression executor, `RKContext`, rather than a generic path that recursively called an older
  binary-only emitter.
- Made the FP16 output boundary accept the canonical FP16-backed `BOOL_MASK` and `INT_FP16` layouts. This closed the one
  physical-ABI gap exposed when BOOL-to-INT output conversion moved from `lower_ew()` to `RKContext`.
- Enabled compact local/RANGE codegen on the default renderer with a 16-byte local-state limit and removed the separate
  `RockchipBoolRenderer` mode. Added `SPECIAL` to generic static materialization so stacked multi-local outputs such as
  `std_mean` compose through their ordinary lane-id UOp.
- Profiled the historical slowest case, `test_nll_loss_3d`, with cProfile and Tinygrad PROFILE events. The old default
  spent 55.54s in its largest `do_to_program`, including 21.72s in Rockchip rendering, then executed 149,903 physical
  EW stages. Compact local codegen reduced the test call from 83.72s to 42.26s, a 49.5% improvement.
- Deleted the 195-line vectorized-unrolled ADD reduction path. Direct unrolled UOp programs now pass through the same
  executor, and the unit contracts check accepted/serializable images instead of requiring that removed optimization's
  gather layout.
- Tested deletion of the remaining 108-line vectorized MUL+ADD residual reducer. A biased two-layer convolution exposed
  a real 0.0127 precision regression, so that path and its strict unit contract were restored. It remains the next
  precision recipe to absorb into the generic RANGE reducer; no tolerance was weakened.
- Removed obsolete `ROCKCHIP_UOPS=0` branches and task-count assertions from the authoritative tests. Submission counts
  now describe only the permanent generic renderer.
- Reduced the renderer from 6,658 to 6,159 executable lines in this milestone, another 499 lines. From the 10,233-line
  pre-deletion baseline, 4,074 executable lines are gone (39.8%). Runtime remains 488 executable lines. The physical
  renderer diff for this milestone is 17 insertions and 526 deletions.
- Stopped repeating the 23-minute full replay at the user's request. The attempted run was clean through 114 passed,
  6 expected skips, and 96 subtests before interruption; focused numerical gates below cover the changed executor,
  local reductions, product precision, and convolution surfaces. A full `/445` replay remains required at the next
  large checkpoint.

Verification:

- Profiled strict `TestRockchipLossOps::test_nll_loss_3d`: 1 passed in 42.26s call time after the change, versus 83.72s.
- Strict `TestRockchipReductionOps`: 36 passed in 134.98s.
- Strict `TestRockchipConvOps`: 42 passed, 6 skipped, and 37 subtests passed in 311.48s.
- Strict dot/einsum plus NLL3D/std-mean/variance precision gate: 16 passed in 115.18s.
- Strict RKContext-only dot/reduction/cast/classification/comparison/logical/bool/WHERE gate: 67 passed in 145.29s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 45. Reject non-index DAGs early and delete dynamic accumulation recovery — complete

- Profiled the new slowest case, `test_nested_conv2d`. Of 186.48s call time under cProfile, 172.51s was spent in
  `_flatten_binary`: the dynamic-index accumulation recognizer recursively visited an ordinary shared convolution ADD
  DAG 97.55 million times before discovering that no runtime index tensor existed.
- Added a linear-time semantic precondition before flattening: the candidate must contain both a `WHERE` and a runtime
  INT `LOAD`. This is a guard on the address-selection rule, not a convolution fast path. `nested_conv2d` fell from
  150.42s to 5.47s call time (96.4% faster), and `large_input_conv2d` fell from 100.05s to 67.15s.
- Removed the dynamic-index accumulation entry points entirely after the performance diagnosis. Deleted the unrolled
  selector recovery, the local-loop selector recovery, affine-load parsing, and three custom accumulation images.
  Direct runtime addresses continue through host gather; expanded selector graphs execute their ordinary UOps.
- The pool/unpool, fancy-index, cumulative-extrema, arg-extrema, sort, and top-k hardware gate passes without the custom
  accumulation path. This confirms that the deleted graph dialect was an optimization/catalog layer, not required
  semantics.
- Reduced the renderer from 6,159 to 5,918 executable lines, another 241 lines. From the 10,233-line pre-deletion
  baseline, 4,315 executable lines are gone (42.2%). The physical renderer diff is 1 insertion and 260 deletions;
  runtime remains 488 executable lines.

Verification:

- Strict `TestRockchipConvOps::test_nested_conv2d`: 1 passed in 5.47s call time, versus 150.42s before the guard.
- Strict `TestRockchipConvOps::test_large_input_conv2d`: 1 passed in 67.15s call time, versus 100.05s.
- Strict pool/unpool/gather/fancy-index/cumulative-extrema/arg-extrema/sort/top-k gate:
  40 passed and 33 subtests passed in 424.82s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 46. Delete superseded unrolled integer selector catalog — complete

- Removed seven unrolled integer/bool recovery handlers and their private byte-equality matrix representation. Generic
  UOp replay now owns bool/int prefix counts and occurrence histograms instead of rediscovering those tensor programs.
- Kept only the two structural address materializers still required by `nonzero`: bounded INT32 lookup and bounded
  integer predicate coordinates. Removing either was tested independently and rejected by the corresponding strict
  `nonzero` case; this is the current semantic boundary rather than dead catalog retention.
- The deletion passed arg-extrema, sort, top-k, INT16 EW, cumulative ADD/MUL/extrema, WHERE, and both nonzero hardware
  gates. No host numeric evaluator or NumPy fallback was added: host participation remains address calculation and raw
  gather/scatter copies only.
- Reduced the renderer from 5,918 to 5,638 executable lines, another 280 lines. From the 10,233-line pre-deletion
  baseline, 4,595 executable lines are gone (44.9%). The physical renderer diff is 2 insertions and 300 deletions;
  runtime remains 488 executable lines.
- The focused duration report identifies cumulative extrema as the next compile/execution bottleneck: simple cummin
  took 90.52s and simple cummax 69.43s. The next milestone will profile that generic structural path before deleting
  another catalog block.

Verification:

- Strict arg-extrema/sort/top-k gate: 6 passed in 39.76s.
- Strict `TestRockchipNonzeroOps`: 2 passed in 49.07s.
- Strict INT16 EW/cumulative ADD/MUL/extrema/WHERE gate: 25 passed in 233.02s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 47. Vectorize static UOp materialization — complete

- Profiled the slowest known case, `test_max_unpool2d`, under cProfile. Of 190.52s total instrumented time, 171.80s
  was `_static_values`; scalar static interpretation made 47.71 million cast calls and 73.92 million recursive UOp
  evaluations. The hardware path itself issued 2,407 ioctls and spent only about 2s inside `ioctl`.
- Reused the existing typed vector UOp evaluator for compile-time RANGE/SPECIAL/index expressions. Static destination
  and value DAGs are now evaluated once over all lanes with shared subexpression caching, then validated and reordered
  exactly as before. This is address/layout planning only; runtime tensor arithmetic remains on the NPU.
- Added a unit contract covering multidimensional RANGE evaluation, WHERE, integer arithmetic, and nontrivial output
  reordering. The wider gather/fancy-index/nonzero/arg-extrema/sort/top-k/WHERE hardware gate also passes.
- `test_max_unpool2d` fell from about 108s to 20.58s (81% faster). The same generic change reduced simple cummin from
  90.52s to 60.42s and simple cummax from 69.43s to 46.84s, both about one third faster.
- This performance milestone adds three executable renderer lines, taking the renderer from 5,638 to 5,641 lines.
  The next milestone resumes deletion from this baseline.

Verification:

- Strict `TestRockchipMaxUnpoolOps::test_max_unpool2d`: 1 passed in 20.58s call time.
- Strict max-unpool plus simple cummin/cummax gate: 4 passed in 131.12s.
- Strict gather/nonzero/fancy-index/arg-extrema/sort/top-k/WHERE gate: 22 passed in 138.67s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 85 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 48. Delete prefix and occurrence graph recovery — complete

- Disconnected and replayed the seven remaining prefix/count/histogram dispatch entries. Generic UOp execution passed
  cumulative ADD/MUL/extrema, one-hot, masked-select, nonzero, arg-extrema, sort, and top-k without those recognizers.
- Deleted the complete now-unreachable 16-function dependency component: scalar local-loop recovery, unrolled FP16 and
  INT32 prefix builders, predicate masks, prefix row materializers, and the INT32 occurrence image. This removes another
  tensor-graph dialect rather than preserving it as an optimization catalog.
- Closed the one generic gap exposed by a unit-only normalized INT32 prefix: comparisons now normalize weak integer
  constants to typed integer operands. The program then composes through ordinary INT32 compare/WHERE UOps. Updated
  the two unit contracts to require accepted, serializable generic images instead of obsolete physical stage counts.
- No CPU numeric semantics were introduced. Compile-time static address/layout evaluation remains the only vectorized
  host calculation; dynamic arithmetic, comparison, WHERE, prefix, and reduction semantics execute as NPU stages.
- Reduced the renderer from 5,641 to 5,254 executable lines, another 387 lines. From the 10,233-line pre-deletion
  baseline, 4,979 executable lines are gone (48.7%). The physical renderer diff is 5 insertions and 420 deletions;
  runtime remains 488 executable lines.
- The generic path was also faster for several formerly recovered programs: simple cumprod fell from 23.32s to 12.12s
  and logcumsumexp from 19.15s to 5.74s in the focused duration reports.

Verification:

- Strict cumulative ADD/MUL/extrema gate: 18 passed in 147.45s.
- Strict one-hot/masked-select/nonzero/arg-extrema/sort/top-k gate: 11 passed in 102.52s.
- Post-gap-fix strict cumsum/simple-cumsum/masked-select gate: 3 passed in 31.42s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 85 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 49. Delete scalar loop boolean reduction recovery — complete

- Disconnected both pre-generic boolean reduction handlers. Ordinary scalar/axis `any` and `all` passed through generic
  RANGE/local UOps, proving the scalar loop recognizer and its integer-predicate image were superseded.
- Retained the grouped boolean structural executor after `test_all_large` demonstrated its distinct SPECIAL/BARRIER/IF
  program is not yet accepted generically. This preserves a proven semantic boundary instead of deleting required code.
- Removed the stale `ROCKCHIP:BOOL` test selector left after milestone 44 deleted the alternate renderer. The large test
  now exercises the sole canonical Rockchip renderer and passes through the retained grouped executor.
- Reduced the renderer from 5,254 to 5,208 executable lines, another 46 lines. From the 10,233-line pre-deletion
  baseline, 5,025 executable lines are gone (49.1%). The physical renderer diff is 2 insertions and 50 deletions;
  runtime remains 488 executable lines.

Verification:

- Strict logical-predicate and boolean-reduction gate: 11 passed in 13.54s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 85 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 50. Share static gather planning across reduction operands — complete

- Profiled the slowest recorded remaining case, `test_large_input_conv2d`. Of 112.08s instrumented time, 93.64s was
  the residual-preserving MUL+ADD reducer and 87.42s was `_gather_offsets`. Its 320 non-affine plans rebuilt the same
  90,720-lane output RANGE environment, causing 174.18 million generator calls.
- Added one lazy `RKStaticIndexEvaluator` per related gather-plan set. It materializes static output RANGE vectors and
  destination lanes once, while each operand retains a fresh expression cache to bound memory. The evaluator only
  handles compile-time INDEX/gate expressions; it never reads or computes runtime tensor values.
- Reused the evaluator for residual MUL+ADD operands and optional bias. Added a unit contract proving two distinct
  gather rows invoke RANGE materialization exactly once and retain correct nontrivial output ordering.
- `test_large_input_conv2d` fell from 67.15s to 18.34–18.62s, about 72% faster. Big GEMM, nested convolution, and biased
  convolution also retain strict numerical output through the same residual-preserving path.
- This performance milestone adds 14 executable renderer lines, taking the renderer from 5,208 to 5,222 lines. The
  next milestone resumes catalog deletion from this baseline.

Verification:

- Strict large-input convolution: 1 passed in 18.62s call time.
- Strict big-GEMM/nested/biased/large-input reduction gate: 4 passed in 31.69s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 86 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 51. Delete bool-total dynamic-load recovery — complete

- Tested deletion of the 147-line dependent scalar-local extrema recognizer. All 27 arg-extrema/sort/top-k/max-pool/
  cumulative-extrema methods remained numerically correct, but `softmax_argmax` regressed from about 3.4s to 156.11s.
  Restored that handler as a measured temporary physical optimization until generic multi-local execution is bounded.
- Independently disconnected all three dynamic-load graph recognizers. Generic `RKHostAddress` handled ordinary typed
  runtime loads, but strict SPECIAL-lane negative-normalized and multi-axis fancy-index programs proved the direct and
  multi-index structural parsers are still required; both were restored.
- Masked-select and nonzero passed without the bool-total-gated dynamic-load recognizer. Deleted that handler and its
  private bool-count parser. Those programs now compose through generic UOps/address materialization instead of a
  specialized `lane < sum(bool)` graph dialect.
- No CPU numeric evaluator was introduced. Host address execution remains limited to reading indices, calculating
  addresses, and copying raw typed bytes; predicate totals and selection semantics remain NPU UOps.
- Reduced the renderer from 5,222 to 5,140 executable lines, another 82 lines. From the 10,233-line pre-deletion
  baseline, 5,093 executable lines are gone (49.8%). The physical renderer diff is 88 deletions; runtime remains 488
  executable lines.

Verification:

- Trial strict arg-extrema/sort/top-k/max-pool/cumulative-extrema gate: 27 passed and 33 subtests passed in 312.10s.
- Final strict one-hot/gather/masked-select/nonzero/fancy-index/scatter gate: 21 passed in 137.16s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 86 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 52. Materialize masked loads before neutral rewrites — complete

- Profiled the current slowest simple cumulative-extrema case, `test_simple_cummin`. The instrumented 101.71s run
  spent 42.79s executing 66,231 ioctls; 10,738 synchronized gather calls consumed 12.68s, while static-value recovery
  consumed another 14.36s. The value and index images contained 2,044 and 1,537 synchronization points respectively.
- Moved generic masked-LOAD materialization ahead of the nonfinite reduction-neutral rewrite, and taught that generic
  fold to commute a finite nonzero scalar multiply through the mask. This preserves the LOAD default directly instead
  of expanding the selector into thousands of alternating gather/EW stages.
- The 512-lane value image fell from 8,177 to 1,030 EW stages and from 2,044 to 2 synchronization points. Strict
  `test_simple_cummin` fell from 58.38s to 42.16s (28% faster), while `test_simple_cummax` fell from 47.36s to 40.24s.
- A trial that forced large FP16 equality chains entirely onto the DPU removed another 4,098 materializations but took
  longer than two minutes, so it was reverted. The retained change addresses the measured bottleneck without trading
  synchronization for a slower physical recipe.
- No CPU numeric semantics were introduced. The host still performs only static index/default materialization; the
  multiply, comparison, selection, and cumulative extrema remain NPU execution.
- This performance milestone adds 16 executable renderer lines, taking the renderer from 5,140 to 5,156 lines. The
  following milestone resumes deletion from this measured baseline; runtime remains 488 executable lines.

Verification:

- Strict simple cumulative minimum: 1 passed in 42.16s call time.
- Strict simple cumulative maximum: 1 passed in 40.24s call time.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 87 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 53. Delete scalar loop reduction recovery — complete

- Disconnected the multi-scalar-local reduction recognizer first. `std_mean` then expanded a 13,125-iteration local
  expression and hit Python recursion depth, so that bounded multi-accumulator structural executor remains required.
- Also trialed deletion of the exact INT32 bounds-mask executor. Thirteen gather/fancy-index/validation cases passed,
  but the extended gate stopped at cold `masked_select(size=4)`, so it was restored conservatively. Later diagnosis
  showed that rejection was the independent masked-LOAD rewrite issue below; bounds-mask deletion remains a candidate.
- Disconnected the ordinary scalar ADD/MUL/MAX loop recognizer. The generic mapped/local UOp paths passed the complete
  reduction census numerically. The sole initial regression was `test_sum_full` using two submit segments instead of
  the required one for its 16,384-lane NOOPT sum.
- Added a physical scheduling peephole that composes a leading idempotent vector materialization with its dependent
  lane gathers. It proves the removed vector is dead before redirecting those gathers to the original operand, so the
  scalar EW chain starts without a synchronization split. This is a physical RKImage rule, not a tensor-op recognizer.
- Deleted the scalar loop parser/executor's 40-line lowering body, its private 42-line reduction-image builder, and the
  now-unreferenced spaced-gather and post-reduction sqrt/cuberoot helpers. SQRT/cuberoot semantics remain owned by the
  generic UOp math recipes.
- A cold-cache masked-select replay exposed a pre-existing rewrite issue hidden by cached images: an outer predicate
  total was being merged into a dynamic LOAD gate. Generic masked-LOAD folding now declines when the outer predicate
  has additional runtime LOAD dependencies, preserving that total as ordinary NPU WHERE/comparison execution.
- No CPU numeric semantics were introduced. The new peephole changes only static gather scheduling, and runtime host
  address work remains limited to reading indices, calculating bounded addresses, and copying raw typed lanes.
- Reduced the renderer from 5,156 to 5,062 executable lines, another 94 lines. From the 10,233-line pre-deletion
  baseline, 5,171 executable lines are gone (50.5%). The physical renderer diff is 32 insertions and 134 deletions;
  runtime remains 488 executable lines.

Verification:

- Cold full `TestRockchipReductionOps`: 36 passed in 121.55s.
- Cold one-hot/masked-select/nonzero gate: 5 passed in 73.60s.
- Cold `test_sum_full`: one NPU submit contract restored; passed in 2.54s call time.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 88 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 54. Narrow the residual loop matcher to dot semantics — complete

- Retried the 60-line INT32 bounds-mask deletion after fixing cold masked-LOAD folding. Cold masked-select/nonzero and
  all renderer units passed without it, but multi-axis integer fancy indexing still rejected while consuming its
  boolean bounds buffer. Restored the exact DPU byte-mask executor; replacing it with host comparison would violate
  the no-CPU-numeric boundary.
- Milestone 53 left `RKLoopReduction` with only one caller: the dot-product physical path. Removed the now-dead scalar
  reduction fields and parsing for generic nodes/environments, post-scale, sqrt, reciprocal, cuberoot, and clamping.
  The matcher now accepts only a plain FP16 local accumulator result and returns the six values dot lowering consumes.
- This is dead-state deletion rather than a new graph dialect. Non-dot reductions fall directly through to the generic
  mapped/local UOp executor, while dot retains its existing product-precision recipe.
- No CPU numeric semantics or host data evaluation were introduced.
- Reduced the renderer from 5,062 to 5,041 executable lines, another 21 lines. From the 10,233-line pre-deletion
  baseline, 5,192 executable lines are gone (50.7%). The physical renderer diff is 8 insertions and 29 deletions;
  runtime remains 488 executable lines.

Verification:

- Cold complete dot/einsum gate: 13 passed in 24.24s.
- Cold std/normalize/full-sum fallthrough gate: 3 passed in 36.46s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 88 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 55. Separate direct-NPU and generic masked-load folding — complete

- A final cold-cache replay after milestone 54 found that the safety rule for generic masked LOADs also disabled the
  existing all-DPU direct multi-index recognizer. Multi-axis integer fancy indexing therefore rejected even though its
  exact INT32 bounds-mask producer had correctly been retained.
- Made additional runtime gate dependencies an explicit opt-in used only by the direct typed-load recognizer. That
  recognizer executes the merged equality/bounds selection on DPU; the generic host-address path continues to keep
  predicate-total gates as ordinary NPU WHERE/comparison UOps.
- Cold multi-axis fancy indexing and cold masked-select now pass together, proving the two ownership paths no longer
  depend on cached compiler images.
- No CPU numeric semantics were introduced. This correctness follow-up adds one executable renderer line, taking the
  renderer from 5,041 to 5,042 lines; runtime remains 488 executable lines.

Verification:

- Cold complete `TestRockchipFancyIndexOps`: 10 passed in 66.17s.
- Cold conflicting fancy-index/masked-select pair: 2 passed in 8.49s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 88 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 56. Keep comparison-derived integer sums in native INT16 — complete

- Profiled the slowest current cold test, `TestRockchipNonzeroOps::test_nonzero`. Its instrumented 80.06s call made
  69.39 million function calls: compilation consumed 38.68s, Rockchip rendering 20.46s, and runtime 35.04s. The
  generated programs contained 9,070 synchronized gather points, 46,213 ioctls, and 36,784 buffer synchronizations.
- Split all nine fixtures and inspected every serialized image. The 32x10 and 10x5x3 cases dominated at 34.24s and
  21.41s in the instrumented diagnostic. Their unrolled prefix images repeatedly converted native `BOOL_INT16`
  comparison results into `INT_FP16` through raw-byte WHERE recipes before adding them.
- Made the physical ABI compositional at the program boundary: when a bounded integer result contains native FP16
  comparisons, choose canonical INT16 scratch instead of INT_FP16. Boolean casts then remain their existing exact
  0/1 INT16 lanes, ADD stays native INT16 DPU execution, and only the terminal output widens to INT32.
- The first two prefix images fell from 158 total synchronization points to 42 and from 438 mid-gathers to 84.
  Cold `test_nonzero` fell from 47.93s to 35.00s, about 27% faster.
- Trialed a more aggressive per-UOp INT16 choice inside mixed INT32 programs. A large coordinate ADD correctly exposed
  that its bounded WHERE operands had already materialized as INT32, so the over-broad experiment was removed rather
  than inserting implicit reinterpretation or host arithmetic.
- No CPU numeric semantics were introduced. Comparison, cast, prefix ADD, coordinate selection, and terminal widening
  all execute on DPU. This performance milestone adds one executable renderer line, taking it from 5,042 to 5,043;
  runtime remains 488 executable lines.

Verification:

- Cold full `TestRockchipNonzeroOps::test_nonzero`: 1 passed in 35.00s call time.
- Cold nonzero-size/masked-select/one-hot gate: 4 passed in 22.16s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 89 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 57. Delete direct integer-to-FP32 cast recovery — complete

- Disconnected the 36-line direct INT32/bool-to-FP32 image. The typed executor already owned INT32-to-FP16 narrowing
  and FP16-to-FP32 output conversion; composing those two physical value boundaries passed the cold cast census.
- Added generic CAST ownership for INT32 values and native BOOL_INT16 values. Boolean conversion remains an ordinary
  DPU WHERE selecting FP16 zero or one, followed by the same terminal FP32 conversion; no host value inspection is
  involved.
- Extended unit coverage to reversed/remapped INT32 and boolean loads, proving the generic path is not limited to a
  contiguous direct-load spelling. The generic INT32 image uses four EW stages where the deleted image used six.
- Removed the complete now-unreachable direct cast function and dispatch. Reduced the renderer from 5,043 to 5,014
  executable lines, another 29 lines. From the 10,233-line pre-deletion baseline, 5,219 executable lines are gone
  (51.0%). The physical renderer diff is 6 insertions and 38 deletions; runtime remains 488 executable lines.
- No CPU numeric semantics were introduced.

Verification:

- Cold complete `TestRockchipCastOps`: 2 passed in 5.84s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 90 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 58. Delete direct FP16-to-INT32 cast recovery — complete

- Disconnected the remaining seven-line direct FP16-to-INT32 wrapper. The first generic replay exposed that a root
  cast from FP16 has no statically provable integer range, so the context selected INT32 before reaching its existing
  half-valued truncation recipe.
- Made the physical ABI rule explicit: an integer program rooted in an embedded FP16 cast uses INT_FP16 scratch unless
  a true dynamic INT32 load already requires INT32. Generic CAST then executes the ordinary DPU truncation recipe and
  the existing terminal INT32 widening.
- Deleted the wrapper and dispatch after cold cast, integral rounding, modulo, and division-rounding tests passed.
  Reduced the renderer from 5,014 to 5,009 executable lines, another 5 lines; runtime remains 488 executable lines.
- No CPU numeric semantics were introduced.

Verification:

- Cold complete `TestRockchipCastOps`: 2 passed in 5.24s.
- Cold integral-rounding/modulo/division-rounding gate: 9 passed in 29.31s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 90 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 59. Group synchronized gathers once per execution point — complete

- Continued the `TestRockchipNonzeroOps::test_nonzero` profile from milestone 56. Runtime line-level profiling found
  9.60s spent rebuilding each synchronized-gather batch by rescanning every mid-gather for every one of its 9,070
  execution points.
- Replaced that quadratic scheduling pass with one stable grouping pass keyed by execution point. Gather order within
  every point, DPU stage boundaries, synchronization, and raw memory movement are unchanged.
- Cold `test_nonzero` fell from 35.00s after the physical-ABI improvement to 28.34s. Relative to its original 47.93s
  cold baseline, the two profiled fixes reduce call time by about 41%.
- The gather-heavy 512/1022-element cumulative-minimum fixture passes cold, exercising the multi-point scheduler over
  large prefix programs.
- No CPU numeric semantics were introduced. This scheduling optimization adds one executable runtime line, taking it
  from 488 to 489; the renderer remains 5,009 executable lines.

Verification:

- Cold full `TestRockchipNonzeroOps::test_nonzero`: 1 passed in 28.34s call time.
- Cold `TestRockchipCumulativeExtremaOps::test_simple_cummin`: 1 passed in 207.61s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 90 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 60. Delete the uint8 cast image catalog — complete

- Disconnected the mapped ADD-loop recognizer first. Small mean/variance/std axis cases passed through the literal
  executor, but full 15x25x35 variance and std reached a roughly 13k-deep expanded accumulator and hit Python's
  recursion limit. Restored that block unchanged: it remains the generic large static ADD-loop executor, not dead
  operation-specific code.
- Moved uint8 into the physical value ABI instead. A half-to-uint8 `CAST` now emits its truncation/modulo recipe through
  ordinary typed UOps, materializes canonical INT16 scratch, and exposes the low byte at the terminal STORE boundary.
  A uint8 `WHERE` composes the same INT16 values and canonical constants without recognizing the surrounding graph.
- Deleted both `_lower_fp16_uint8_cast` and `_typed_int16_byte_image`, plus their early dispatch. Renderer executable
  size fell from 5,009 to 4,984 lines, another 25 lines. From the
  10,233-line pre-deletion baseline, 5,249 executable lines are gone (51.3%); runtime remains 489 lines.
- No CPU numeric semantics were introduced. Truncation, modulo, WHERE, and FP16-to-INT16 conversion all execute on DPU;
  the terminal gather only materializes the selected low byte.

Verification:

- Cold complete `TestRockchipCastOps`: 2 passed in 6.10s.
- Cold upstream `TestOpsUint8::test_cast`: 1 passed in 2.74s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 61. Delete the boolean cast image wrapper — complete

- Moved FP16-to-bool `CAST` into the typed UOp handler. It now composes the existing native ABS and positive-mask
  recipes for any FP16 expression, producing the canonical BOOL_MASK physical value before the normal terminal bool
  conversion.
- Deleted `_typed_int_image` and the early direct-load boolean dispatch. The old wrapper rebuilt the output store and
  retargeted the terminal image; the generic context already owns both scratch placement and bool output conversion.
- Added coverage for a reversed, composed `(load + 1).cast(bool)` program, not only the direct-load spelling. The same
  expression passes on hardware for zeros, signs, infinities, and NaN.
- Renderer executable size fell from 4,984 to 4,971 lines, another 13 lines. From the 10,233-line pre-deletion
  baseline, 5,262 executable lines are gone (51.4%); runtime remains 489 lines.
- No CPU numeric semantics were introduced. ABS, comparison-mask generation, and bool output conversion execute on
  DPU.

Verification:

- Cold `TestRockchipCastOps::test_cast`: 1 passed in 6.29s.
- Composed FP16-expression-to-bool hardware boundary check: pass.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 62. Restore FP16 state without resetting after native INT16 chains — complete

- Profiled the current slowest case, the 512-element half of `test_simple_cummin`. Its instrumented 89.69s execution
  spent 55.03s in ioctl and 54.01s specifically in 513 NPU resets. The cumulative-index image alternates long native
  INT16 byte/mask chains with one ordinary FP16 MAX at each structural gather point.
- Replaced reset-on-INT16-to-FP16 transition with a fully stateful FP16 stage. The emitted DPU operation, dependency
  order, gathers, submits, and tasks are unchanged; the stateful stage explicitly restores the ordinary FP16 register
  configuration that the reset previously supplied.
- A cold standalone 512-element cumulative minimum still executes exactly 2,052 submits and 24,158 DPU tasks, but now
  completes in 12.65s. The complete 512/1,022-element `test_simple_cummin` fell from 207.61s to 28.77s, about 86% faster.
- The full cumulative-extrema, comparison, and cast hardware gate passes, covering native INT16, ordinary FP16,
  comparison isolation, terminal integer conversion, and subsequent-program state recovery.
- No CPU numeric semantics were introduced. This runtime scheduling change adds two executable lines, taking runtime
  from 489 to 491; renderer remains 4,971 executable lines.

Verification:

- Cold `TestRockchipCumulativeExtremaOps::test_simple_cummin`: 1 passed in 28.77s.
- Cold cumulative-extrema/comparison/cast gate: 15 passed in 80.51s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 63. Delete the dot-loop graph recognizer — complete

- Disconnected the remaining dot-loop fast path so dot, matvec, multidot, broadcast-dot, and einsum programs had to
  use the ordinary mapped static ADD reduction. The complete cold dot/einsum gate passed before deletion, proving the
  private graph parser was no longer required for correctness or precision.
- Deleted `_loop_reduction_shape`, `RKLoopReduction`, `_loop_reduction_match`, `_lower_dot_loop_reduction`, and the now
  unreachable `_lower_composed_uops` wrapper. This removes tensor-program recovery of MUL+ADD loops; the surviving
  structural executor consumes RANGE/local LOAD/STORE/ADD semantics generically.
- Renderer executable size fell from 4,971 to 4,910 lines, another 61 lines. From the 10,233-line pre-deletion
  baseline, 5,323 executable lines are gone (52.0%). The physical renderer diff is 70 deletions; runtime remains 491
  executable lines.
- No CPU numeric semantics were introduced. Products and reductions remain DPU EW stages, and no WMMA/matmul graph
  recognition replaced the deleted path.

Verification:

- Cold complete `TestRockchipDotOps` and `TestRockchipEinsumOps`: 13 passed in 22.34s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 64. Delete stale physical-value and rejection scaffolding — complete

- Removed the unused `RKMultiGather` and `RKStatic` placeholder types and their `RKLeaf` aliases. No live lowering,
  serialization, runtime, or test path referenced them; physical values are represented directly by the current typed
  `RKValue`/`RKArg` machinery.
- Removed two diagnostic-only `reject` closures from the generic mapped and multi-local reduction executors. Their
  reason strings were never observed, so each branch now returns the same `None` result directly. Also deleted two
  unreferenced EW/pool constants.
- Renderer executable size fell from 4,910 to 4,900 lines. From the 10,233-line pre-deletion baseline, 5,333
  executable lines are gone (52.1%); runtime remains 491 executable lines.
- No CPU numeric semantics or lowering behavior were introduced. This milestone is semantics-neutral dead-code
  removal, so it does not require the currently unavailable NPU for behavioral validation.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 65. Delete dormant reduction-helper modes — complete

- Audited all repository callers of the private row and arena reducers, equality-mask builder, scalar evaluator, and
  output-store parser. Deleted configuration state that no caller supplied: INT32 row reduction, FP32/INT16 arena
  output, per-level arena barriers, configurable equality barriers, scalar-evaluator cache injection, and explicit
  REDUCE rejection.
- Kept the only live physical behavior unchanged: FP16/INT16 balanced rows, optional per-operation barriers for
  precision reducers, and the fixed SUB/ABS/compare equality sequence. Removed the final pool-index constant that
  became unreferenced with the stale radix in milestone 64.
- Renderer executable size fell from 4,900 to 4,891 lines. From the 10,233-line pre-deletion baseline, 5,342
  executable lines are gone (52.2%); runtime remains 491 executable lines.
- No CPU numeric semantics or lowering behavior were introduced. This is a private dead-mode deletion validated by
  serialized-image unit contracts and static checks; hardware remains unavailable pending reboot.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 66. Delete the unreachable fill ABI — complete

- Audited every `RKImage` producer across the repository. No renderer path constructed `RKFill`; only the decoder
  could recreate it, leaving its serializer flag, scratch-lifetime handling, append guards, and runtime host memcpy
  branch unreachable from any compiled UOp program.
- Deleted `RKFill` end to end and reserved its former header flag. Bumped `RKIMAGE_VERSION` from 31 to 32 so any stale
  serialized blob is rejected explicitly rather than interpreted under the narrower physical ABI.
- Renderer executable size fell from 4,891 to 4,880 lines and runtime fell from 491 to 486 lines. From the 10,233-line
  pre-deletion renderer baseline, 5,353 executable lines are gone (52.3%).
- No CPU numeric semantics or generated execution behavior changed. Compile-time constants continue through ordinary
  gather materialization; this removes a host numeric-fill path that the UOp renderer never emitted.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed, including all RKImage round trips.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 67. Remove fixed target, version, and alignment state — complete

- Audited every image and scratch producer. All images target RK3588, every decoder rejects versions other than the
  current one, and every scratch allocation uses 4 KiB alignment. These values were carried as per-object state even
  though no producer could vary them and neither target nor version affected runtime execution.
- Removed `RKTarget`, `RKImage.target`, `RKImage.version`, and `RKScratch.alignment`. The serializer now writes its
  compile-time image version directly, scratch records contain only their variable size, and the runtime applies the
  sole 4 KiB scratch-alignment contract directly. Bumped the serialized ABI to version 33.
- Renderer executable size fell from 4,880 to 4,878 lines; runtime remains 486 lines. From the 10,233-line renderer
  baseline, 5,355 executable lines are gone (52.3%). More importantly, 37 physical lines and three false ABI degrees
  of freedom were removed.
- No CPU numeric semantics or generated DPU task changed. Serialized-image round trips cover the narrowed records.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 68. Delete the duplicate image gather split — complete

- Every mid-program gather already serializes its own physical `after` stage, but `RKImage.gather_after` retained a
  second fallback split in the image header, scratch coloring, image composition, and runtime scheduler. Audited all
  producers and made the few legacy mid-gather constructors write their existing split directly into `RKGather.after`.
- Deleted `RKImage.gather_after` and the corresponding header/runtime fallback logic. Image composition now offsets
  each explicit point, and decode validates every mid-gather point directly. Bumped the serialized ABI to version 34.
- One unit contract had asserted a `gather_after` value even though its image contained no mid-gathers; replacing that
  meaningless state assertion with the actual emitted EW/output contract confirmed the field was not semantic.
- Renderer executable size fell from 4,878 to 4,876 lines; runtime remains 486 lines. From the 10,233-line renderer
  baseline, 5,357 executable lines are gone (52.3%). No CPU numeric semantics or DPU operation changed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 69. Delete the unconstructible host negative-index flag — complete

- Audited every `RKHostAddress` constructor. None enabled `normalize_negative`; negative-index normalization that is
  semantically present in Tinygrad is either executed by the native typed-load path or already represented in the
  address UOps before host materialization. The serialized flag and runtime NumPy branch were therefore unreachable.
- Removed `RKHostAddress.normalize_negative`, its image-record word, decoder validation, and runtime branch. Bumped the
  serialized ABI to version 35.
- Renderer executable size fell from 4,876 to 4,874 lines and runtime fell from 486 to 485 lines. From the 10,233-line
  renderer baseline, 5,359 executable lines are gone (52.4%).
- No CPU numeric semantics were added; this deletes an unused host-side semantic option while retaining bounded raw
  address calculation and movement.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 70. Merge terminal gathers into the physical stage timeline — complete

- `post_gathers` were not a distinct hardware operation: they were raw gathers executed after the final EW stage,
  exactly the point already represented by `RKGather.after == len(ew_ops)`. Moved every specialized and generic image
  producer onto that explicit point and deleted the separate `RKImage` phase.
- Removed the post-gather header count, decoder partition, scratch-coloring pass, image-composition branches, serialized
  field, and runtime submission branch. Existing terminal mid-gathers and former post-gathers now retain their original
  order in one synchronized batch. Bumped the serialized ABI to version 36.
- The unit audit exposed one obsolete distinction: a raw-bitcast image already had two ordinary scratch gathers at the
  final point before its separate output gather. The contract now verifies the terminal ARG movement rather than the
  deleted phase label.
- Renderer executable size fell from 4,874 to 4,869 lines and runtime fell from 485 to 484 lines. From the 10,233-line
  renderer baseline, 5,364 executable lines are gone (52.4%). No CPU numeric semantics or DPU arithmetic changed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed, including all image round trips and
  terminal raw-bit layout contracts.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 71. Unify typed nonzero parsing and remove encoding wrappers — complete

- Built a fresh private-component reference inventory across the renderer and runtime. No remaining large recognizer
  is provably superseded without hardware replay: the grouped boolean, bounded dynamic-address, INT32, and precision
  components each still have a distinct live dispatch or previously recorded failure boundary.
- Replaced the duplicated FP16 and integer `LOAD != 0` graph parsers with one dtype-parameterized UOp matcher. Both
  grouped boolean reduction and FP16 boolean-cast rewriting now consume the same semantic rule.
- Inlined the two remaining single-use static FP16-vector and FP32-bit encoding wrappers at their physical consumers.
  Renderer executable size fell from 4,869 to 4,860 lines. From the 10,233-line baseline, 5,373 executable lines are
  gone (52.5%); runtime remains 484 lines.
- No CPU numeric semantics were introduced. The consolidated matcher only classifies UOps; comparison, reduction, and
  cast execution remain DPU operations.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 72. Unify raw two-byte physical decomposition — complete

- Replaced the separate FP16 and INT16 raw-byte caches and splitters with one `RKValue` physical operation. Both
  layouts occupy the same two-byte lane representation; their semantic distinction remains in `RKValue.layout`, while
  raw movement now depends only on the shared physical width.
- INT16 bitwise operations, FP16 IEEE comparisons, and repacked INT16 results all reuse the same byte cache. This
  removes duplicated scratch allocation, gather scheduling, and cache bookkeeping without recognizing any tensor op.
- Renderer executable size fell from 4,860 to 4,848 lines. From the 10,233-line baseline, 5,385 executable lines are
  gone (52.6%); runtime remains 484 lines.
- No CPU numeric semantics were introduced. Byte splitting and repacking remain raw gather movement, while boolean and
  arithmetic recipes remain DPU execution.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed, including exact FP16 payload/sign
  bitcasts, INT16 masks, byte logic, serialization, and scratch scheduling.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 73. Delete unused UOp use-count registration — complete

- Audited every `RKContext.use_counts` read and write. `_register_graph` walked each root and every generated recipe,
  merged source-use counts into the context, and no allocator, cache, lowering handler, or scheduler ever read them.
- Deleted the counter, registration method, initial root walk, and all 19 recipe-registration calls. Recipe UOps are
  still lowered through the same memoized `values` map; scratch lifetime reuse remains an RKImage scheduling pass and
  never depended on semantic graph use counts.
- Renderer executable size fell from 4,848 to 4,822 lines. From the 10,233-line baseline, 5,411 executable lines are
  gone (52.9%); runtime remains 484 lines. The physical diff is exactly 27 deletions with no replacement code.
- No CPU numeric semantics or generated DPU task changed. The required standalone NPU health check still times out,
  so hardware-sensitive recognizer replay remains pending reboot while static dead-code work continues.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `.venv/bin/python ~/rk3588/examples/elementwise.py`: timed out in `DRM_IOCTL_RKNPU_SUBMIT` (device still needs reboot).

### 74. Collapse the orphaned ALU destination allocator — complete

- Removing graph use counts exposed `_alu_dst` as an exact duplicate of `_dst`: its operand tuple was immediately
  discarded, and both methods applied the same root-output ABI check before allocating scratch.
- Routed NEG and binary ALU handlers through `_dst`, deleted `_alu_dst`, stopped constructing four unused operand
  tuples, and removed the never-read `RKContext.store` field.
- Renderer executable size fell from 4,822 to 4,815 lines. From the 10,233-line baseline, 5,418 executable lines are
  gone (52.9%); runtime remains 484 lines.
- No CPU numeric semantics or generated hardware behavior changed. This deletes residual allocation scaffolding whose
  only former purpose was semantic use-count tracking.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 75. Finish the single gather timeline and static materializer — complete

- Removed the last renderer-internal `terminal_gathers` side list. Final raw output copies now enter the same
  `mid_gathers` timeline with its existing negative terminal sentinel, and `finish()` resolves that sentinel to the
  final EW boundary once the complete physical program is known. Runtime behavior and the serialized ABI are
  unchanged; this completes the phase collapse started in milestone 70.
- Proved `_static_int32()` duplicated `_static()`'s canonical INT32 branch. Its sole caller is an integer comparison
  containing a runtime INT32 load, which already selects `RKLayout.INT32` for the context, so the general static
  materializer creates the identical vector, cache key, scratch slot, and gather. Deleted the duplicate helper.
- Renderer executable size fell from 4,815 to 4,807 lines. From the 10,233-line baseline, 5,426 executable lines are
  gone (53.0%); runtime remains 484 lines. No CPU numeric semantics or DPU arithmetic changed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 76. Delete orphaned index declarations and gather modes — complete

- The whole-repository symbol audit found an unused native `copysign` graph tag and two obsolete dynamic-index tuple
  aliases. None had a producer, consumer, import, or annotation after the earlier recognizer deletions, so all three
  declarations are gone.
- Both runtime gather paths always requested scratch clearing. Removed the unused `clear_scratch` parameters and
  conditional mode from `apply_gathers()` and `synchronized_gathers()`; the one-clear-per-physical-slot behavior is
  unchanged. Also reused the existing checked `buffer()` resolver for DMA addresses and removed a redundant nonempty
  guard after the method's early return.
- Renderer executable size fell from 4,807 to 4,805 lines and runtime fell from 484 to 480 lines. From the 10,233-line
  renderer baseline, 5,428 executable renderer lines are gone (53.0%). No host numeric evaluation or DPU behavior was
  added or changed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 77. Stop recursive compensated-recipe expansion — complete

- Profiled the slowest static lowering test, `test_nested_fp32_product_sum_is_committed_before_outer_half_add`.
  A four-product expression took 3.49 seconds and produced 31,281 EW stages. The profile attributed 2.01 seconds to
  21 `_accurate_add_recipe()` calls, with the same already-compensated storage graph expanded at 19 overlapping ADD
  subgraphs; scratch coloring then spent another 1.83 seconds walking the inflated physical program.
- Added one canonical `_tag_precise_adds()` walker and apply it after half-storage algebra/constant rewriting. Algebraic
  simplification can no longer erase the marker that says a compensated physical recipe is complete, so generic ADD
  lowering executes that recipe rather than recursively compensating it again.
- Reused the same walker for product sums, FP32 ratio correction, math recipes, and physical recipe expansion, deleting
  four duplicated graph-tagging blocks. The profiled expression now emits 103 EW stages and takes 0.42 seconds in the
  full static suite (88% faster); the 91-test suite fell from 7.23 to 6.03 seconds.
- Renderer executable size fell from 4,805 to 4,789 lines. From the 10,233-line baseline, 5,444 executable renderer
  lines are gone (53.2%); runtime remains 480 lines. This changes only compile-time UOp recipe ownership and removes
  redundant DPU work—there is no CPU numeric evaluation.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12 --durations=10`: 91 passed in 6.03 seconds.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py::test_nested_fp32_product_sum_is_committed_before_outer_half_add -q -n0`:
  1 passed in 0.81 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 78. Unify EW command-stage finalization — complete

- Normal FP16 and stateful EW emitters duplicated command packing, three relocation records, relocation placeholder
  commands, the RDMA feature-mode tail, and `RKStage` construction. Moved that shared physical encoding into one
  `_finish_ew_stage()` helper; each emitter now owns only its genuinely different register program and feature value.
- Compared the refactored emitter directly with commit `d0f83926c` across ten normal, FDIV, ReLU6, compare, stateful,
  native INT16, native INT32, INT16→INT32, FP32-input, and FP32-output configurations. Command tuples and relocation
  `(word, kind, index, addend)` records were byte-identical in every case.
- Reused the context's one root topological order for mask detection, integer-layout classification, and static-node
  classification instead of traversing the same root three times.
- Renderer executable size fell from 4,789 to 4,785 lines. From the 10,233-line baseline, 5,448 executable renderer
  lines are gone (53.2%); runtime remains 480 lines. No execution semantics or host numeric work changed.

Verification:

- Previous/current EW encoder comparison: 10/10 command and relocation configurations identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 79. Remove residual recognizer and terminal-copy scaffolding — complete

- A scoped parameter/use audit found that grouped boolean reduction no longer consumed its `uops` argument after its
  structural proof moved entirely onto `RKOutput`. Removed the dead parameter and avoided copying the root's existing
  topological order. Also removed unused output-store and loop-position unpack targets in the vectorized reduction,
  multi-local reduction, and host-scatter paths.
- Three one-item terminal byte-copy tuples survived the gather-timeline migration. Each was created only to run a
  generator that set `after=len(ops)`. Scheduled those byte copies directly at the final EW point and deleted the
  temporary phase-shaped scaffolding.
- Renderer executable size fell from 4,785 to 4,782 lines. From the 10,233-line baseline, 5,451 executable renderer
  lines are gone (53.3%); runtime remains 480 lines. No numeric semantics, memory policy, or hardware schedule changed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.55 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 80. Tag compensated ADDs at construction — complete

- Profiled the new slowest static case, `test_dependent_reduction_range_preserves_vector_output_axis`. Its literal
  65/128-product correctness path emits 4,012/7,918 real EW stages, but `_precise_mul_sum()` then rebuilt that entire
  completed UOp graph solely to attach `_NATIVE_PRECISE_ADD` markers.
- Added one physical `_precise_add()` constructor and use it throughout error-free product/sum construction. Input
  product DAGs are tagged together once before expansion, while every newly generated ADD owns its marker immediately;
  the large completed recipe no longer needs another topological traversal and reconstruction.
- Compared serialized RKImages before and after for scalar 65-term, vector 65-term, large 128-term, and nonaffine
  64-product programs. All four images were byte-identical, including gathers, scratch coloring, EW stages, and flags.
  Large lowering fell from 0.740 to 0.568 seconds (23%), the suite's slowest case fell from 2.21 to 1.59 seconds, and
  the complete static suite now takes 5.29 seconds.
- Renderer executable size fell from 4,782 to 4,781 lines. From the 10,233-line baseline, 5,452 executable renderer
  lines are gone (53.3%); runtime remains 480 lines. This is compile-time graph construction only: no CPU numeric
  semantics and no hardware-stage change.

Verification:

- Old/current `encode_image()` comparison: 4/4 profiled reduction programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.29 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 81. Canonicalize static slots and physical remapping — complete

- Removed two copies of typed static-vector allocation/cache/gather logic. INT32 constants and arbitrary static UOp
  vectors now use one `_static_slot()` physical materializer, while preserving the requested `RKValue.dtype` when two
  semantic dtypes share the same canonical layout and bits.
- Generic half/INT16/INT32 LOAD lowering computed the identical default raw bits independently in its dynamic
  host-address and static-gather branches. Compute that boundary encoding once after float/bool handling and share it;
  host movement remains raw addressing only.
- Reprofiled the remaining slow dependent reduction. Scratch coloring spent 0.612 seconds largely cloning immutable
  dataclasses. Direct `RKArg` and `RKEWOp` reconstruction preserves every field while cutting coloring to 0.475 seconds
  (22%). Compared old/current serialized images for scalar, vector, and large reductions; all were byte-identical.
- The slowest suite case fell from 1.59 to 1.40 seconds and the complete suite from 5.29 to 4.98 seconds. Renderer
  executable size fell from 4,781 to 4,775 lines; from the 10,233-line baseline, 5,458 lines are gone (53.3%). Runtime
  remains 480 lines. No CPU numeric semantics or physical command changes were introduced.

Verification:

- Old/current `encode_image()` comparison: scalar/vector/large profiled reductions byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12 --durations=6`: 91 passed in 4.98 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 82. Unify typed gather-backed materialization — complete

- Bool LOADs, nonaffine dynamic candidate matrices, and ordinary half/INT16/INT32 LOADs each repeated the same
  gather-key, scratch-allocation, destination-patching, and `RKValue` cache block. Added one `_gather_slot()` boundary
  so these semantic LOAD forms share the same physical materialization operation.
- Static vectors and gathered vectors then exposed the same underlying cache/allocate/materialize behavior. Replaced
  separate `static_slots` and `gather_slots` state with one tagged `materialized_slots` cache and one
  `_materialized_slot()` implementation. Static, gather, and FP32-raw keys remain disjoint, and requested semantic
  dtype is preserved when returning a cached physical slot.
- Removed the gather helper's item-size override mode: callers now pass a complete physical `RKGather`, so its cache
  key describes the exact materialization. Default raw bits are likewise computed once for both static and dynamic
  typed LOAD branches.
- Compared milestone-81/current serialized images for FP16 remap, INT32 direct load, bool direct load, nonaffine
  dynamic host-address materialization, and FP32 raw load. All five RKImages were byte-identical.
- Renderer executable size fell from 4,775 to 4,759 lines. From the 10,233-line baseline, 5,474 executable renderer
  lines are gone (53.5%); runtime remains 480 lines. No CPU numeric semantics or DPU schedule changed.

Verification:

- Old/current `encode_image()` comparison: 5/5 representative materialization programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.40 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 83. Skip no-op semantic graph rebuilds — complete

- Reprofiled `test_dependent_reduction_range_preserves_vector_output_axis` after materialization cleanup. The largest
  avoidable semantic cost was `_finite_int_max_neutrals()`: it rebuilt every node of 4k/8k-stage FP16 arithmetic
  graphs even though they contained no `INT32_MIN`, the only constant that pass can change. Added an exact no-op guard.
- Large graphs already disable optional accurate/composite recipes, but `_expand_math_uops()` still rebuilt an 8,696
  node ADD/MUL/load graph containing no transformable math. It now returns the original UOps when a large graph has no
  WHERE, SQRT, EXP2, LOG2, SIN, TRUNC, or float→half storage boundary. All those semantic cases retain the full pass.
- Deleted the renderer's private recursive `_substitute_static_ranges()` graph walker and delegated exact UOp
  replacement to `UOp.substitute()`, Tinygrad's existing bottom-up substitution mechanism. This removes duplicate
  graph infrastructure and retains the generic local/RANGE executor.
- Compared milestone-82/current serialized images for scalar 65-term, vector 65-term, and large 128-term dependent
  reductions; all three were byte-identical. Profiled process time fell from 3.511 to 3.295 seconds, and isolated test
  call time fell from roughly 1.20 to 1.08 seconds.
- Renderer executable size fell from 4,759 to 4,756 lines. From the 10,233-line baseline, 5,477 executable renderer
  lines are gone (53.5%); runtime remains 480 lines. No CPU numeric semantics or generated DPU stages changed.

Verification:

- Old/current `encode_image()` comparison: 3/3 dependent reduction programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12 --durations=6`: 91 passed in 5.16 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 84. Make static UOp substitution iterative — complete

- Audited the one-line `_substitute_static_ranges()` forwarding wrapper. Replacing it outright with generic
  `UOp.substitute()` was functionally correct but profiling rejected it: 264 small substitutions cost 0.362 seconds.
  Rewrote the operation as a compact iterative topological UOp pass, preserving exact replacement semantics without
  recursion or general pattern-rewrite overhead; the same workload now takes 0.034 seconds (91% faster).
- Generic LOAD lowering recomputed the same index and gate topological orders up to twelve times while classifying
  dynamic addressing. Materialize each node order once and share the runtime-address predicate, dynamic-index parser,
  and gate proof. This is graph inspection only; host code still performs no arithmetic semantics.
- Removed unused default dtype modes from three internal specialized entry points; every caller already supplies the
  semantic source dtype explicitly. Kept `_iter_range_env(max_envs=...)` after its direct bounded-allocation contract
  test proved that parameter is live.
- Compared milestone-83/current serialized images for 65-term and 128-term dependent reductions plus nonaffine dynamic
  host-address materialization; all three were byte-identical. Profiled test process time fell from 3.657 to 3.318
  seconds, and the full suite's slowest case fell from 1.51 to 1.41 seconds.
- Renderer executable size fell from 4,756 to 4,754 lines. From the 10,233-line baseline, 5,479 executable renderer
  lines are gone (53.5%); runtime remains 480 lines. No CPU numeric semantics or DPU stage changed.

Verification:

- Old/current `encode_image()` comparison: 3/3 profiled structural/address programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12 --durations=6`: 91 passed in 5.26 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 85. Canonicalize native INT16 EW flags — complete

- Twelve physical recipe blocks independently allocated the same `int16_input=True, int16_output=True` keyword
  bundle. Replaced those local aliases with one shared `_INT16_EW` physical configuration and removed eleven
  executable renderer lines. The recipes retain their true UOp ownership and no tensor-operation dispatch was added.
- Compared milestone-84/current serialized images for IEEE FP16 comparison, raw FP16 bitcast, INT32 bitwise, and
  INT32 shift programs; all four were byte-identical. This is a constructor-only cleanup and changes neither scratch
  layout nor DPU scheduling.
- Renderer executable size fell from 4,754 to 4,743 lines. From the 10,233-line baseline, 5,490 executable renderer
  lines are gone (53.6%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 4/4 representative native-INT16 programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.11 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 86. Stream scratch lifetime spans — complete

- Profiled `test_dependent_reduction_range_preserves_vector_output_axis`, still the suite's slowest case. Scratch
  coloring retained every physical touch in a Python list and then rescanned 16,467 histories with `min()`/`max()`.
  Physical events are visited monotonically, so the allocator now stores only each virtual slot's first and latest
  event and keeps slots in first-touch order. This preserves the same linear-scan coloring order without the sort.
- Precompute remapped zero-offset `RKArg` values once instead of reconstructing the same argument for every source and
  destination occurrence. Nonzero byte offsets retain exact per-use reconstruction.
- The scratch-coloring cProfile cost fell from 0.591 to 0.393 seconds. Alternating the complete scalar/vector/large
  workload between milestone 85 and current code reduced its median from 1.006 to 0.893 seconds (11.2%); the slowest
  full-suite case measured 1.24 seconds, down from 1.40 seconds before profiling.
- Compared milestone-85/current serialized images for scalar, vector, and large dependent reductions; all three were
  byte-identical. Renderer executable size is 4,747 lines (four lines added for the faster lifetime representation),
  still 5,486 lines or 53.6% below the 10,233-line baseline. Runtime remains 480 lines and no CPU numeric semantics
  were introduced.

Verification:

- Old/current `encode_image()` comparison: 3/3 profiled dependent reductions byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12 --durations=3`: 91 passed in 5.05 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 87. Deduplicate storage-boundary recipes — complete

- `_pm_fp32_to_fp16` and `_pm_generic_storage_precision` repeated the same ten FP32/FP16 physical rewrite rules.
  Defined that ordered rule sequence once as `_pm_storage_common`; the full conversion pass adds its bool/WHERE rules,
  while generic storage precision adds its float-WHERE boundary before the shared rules. No semantic UOp or physical
  recipe changed.
- Inlined the single-use binary16 predecessor calculation at LOG2 exponent extraction and removed its wrapper.
- Compared milestone-86/current serialized images for bool-to-half conversion, FP32 storage algebra, float `WHERE`,
  and LOG2; all four were byte-identical.
- Renderer executable size fell from 4,747 to 4,736 lines. From the 10,233-line baseline, 5,497 executable renderer
  lines are gone (53.7%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 4/4 representative storage/math programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.18 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 88. Share repeated math UOp recipe fragments — complete

- SIN, COS, and TAN each rebuilt the same absolute-angle reflection around pi/2. Added one
  `_dpu_reflected_angle()` UOp recipe fragment and reused it in all three handlers.
- Normal and statically masked EXP2 each rebuilt the same bounded fractional polynomial and exact integer-power
  scaling. Added one `_dpu_exp2_bounded()` recipe fragment and retained domain masking in the owning EXP2 handlers.
- Compared milestone-87/current graph keys for half SIN, FP32 additive SIN, COS, TAN, EXP2, and nonpositive EXP2.
  Every generated semantic UOp graph was identical, including 735 nodes in the largest FP32 SIN recipe.
- Renderer executable size fell from 4,736 to 4,726 lines. From the 10,233-line baseline, 5,507 executable renderer
  lines are gone (53.8%); runtime remains 480 lines. These helpers only construct UOps and introduce no CPU numeric
  semantics.

Verification:

- Old/current UOp graph-key comparison: 6/6 math recipes identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.24 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 89. Canonicalize RKContext INT16 scratch allocation — complete

- Five FP16/INT32 comparison methods each declared the same local callback for allocating a canonical INT16 scratch
  argument. Added one `_int16_arg()` physical ABI method on `RKContext` and passed it directly to raw-byte equality,
  ordering, and classification recipes.
- Compared milestone-88/current serialized images for FP16 inequality, FP16 less-than, and INT32 less-than; all three
  were byte-identical.
- Renderer executable size fell from 4,726 to 4,722 lines. From the 10,233-line baseline, 5,511 executable renderer
  lines are gone (53.9%); runtime remains 480 lines. No CPU numeric semantics or DPU schedule changed.

Verification:

- Old/current `encode_image()` comparison: 3/3 comparison programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.39 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 90. Canonicalize standalone scratch arguments — complete

- Eight bounded-index, raw-gather, and boolean-reduction builders each declared the same local `RKArg(SCRATCH, ...)`
  constructor. Replaced them with one module-level `_scratch_arg()` physical helper, including the callback passed to
  the existing equality-mask recipe.
- Reused `_INT16_EW` while touching the long byte-mask constructor block; this only replaces repeated dataclass
  keywords and keeps the encoded operation flags unchanged.
- Compared milestone-89/current serialized images for bounded index selection, dynamic raw gather, matrix boolean
  reduction, and contiguous boolean reduction. All four were byte-identical, including fallback builders not selected
  by the generic-first unit corpus.
- Renderer executable size fell from 4,722 to 4,717 lines. From the 10,233-line baseline, 5,516 executable renderer
  lines are gone (53.9%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 4/4 standalone physical builders byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.30 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 91. Use iterative substitution at the final local boundary — complete

- Reprofiled `test_dependent_reduction_range_preserves_vector_output_axis`. The single-buffer local interpreter's
  final accumulator replacement still called generic `UOp.substitute()` even though the multi-buffer branch and each
  range iteration already used `_substitute_static_ranges()`. That one generic call cost 0.355 seconds in the profile.
- Routed the final replacement through the same iterative topological substitution pass. `_unroll_static_local()`
  fell from 0.619 to 0.261 seconds (58%), and total `_lower_uop_program()` time in cProfile fell from 2.049 to 1.796
  seconds. The alternating direct scalar/vector/large workload median fell from 0.932 to 0.737 seconds (20.9%).
- Compared milestone-90/current serialized images for scalar, vector, and large dependent reductions; all three were
  byte-identical. Renderer size remains 4,717 executable lines, 5,516 lines or 53.9% below baseline; runtime remains
  480 lines. This is graph substitution only and adds no CPU numeric semantics.

Verification:

- Old/current `encode_image()` comparison: 3/3 dependent reductions byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12 --durations=5`: 91 passed in 5.10 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 92. Inline single-use storage callbacks — complete

- Removed the last local `int16_input/int16_output` dictionary and reused `_INT16_EW` in bounded predicate-coordinate
  emission.
- Inlined three one-use matcher callbacks for dynamic FP32 ALU conversion, float `WHERE` conversion, and bool-to-half
  materialization into their owning storage-boundary pattern tables. The matcher order and returned UOps are unchanged;
  these callbacks were not shared semantic APIs.
- Compared milestone-91/current serialized images for bool-to-half conversion, FP32 storage algebra, and float
  `WHERE`; all three were byte-identical.
- Renderer executable size fell from 4,717 to 4,712 lines. From the 10,233-line baseline, 5,521 executable renderer
  lines are gone (54.0%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 3/3 storage-boundary programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.35 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 93. Skip semantic clamp matching for physical ADDs — complete

- Reprofiled the dependent reduction after milestone 91. Its 9,258 compensated half ADDs carry
  `_NATIVE_PRECISE_ADD`, but `_alu()` still attempted the semantic ReLU-cap graph matcher for every one. Restricted
  that matcher to untagged `arg is None` ADDs, which are the only UOps it can rewrite.
- `_fold_relu_cap()` disappeared from the profile, `_alu()` fell from 0.467 to 0.420 seconds, and total
  `_lower_uop_program()` time fell from 1.796 to 1.747 seconds. Scalar, vector, and large dependent-reduction RKImages
  remained byte-identical.
- Inlined the single-use COS and TAN recognition callbacks into their owning pattern entries. Old/current COS and TAN
  rewrite graphs were identical; the UOp math handlers remain unchanged.
- Renderer executable size fell from 4,712 to 4,709 lines. From the 10,233-line baseline, 5,524 executable renderer
  lines are gone (54.0%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 dependent-reduction RKImages byte-identical and 2/2 trig rewrite graphs identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12 --durations=3`: 91 passed in 5.25 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 94. Share masked materialization patterns — complete

- The early masked-load pass and the FP32-to-FP16 storage pass embedded the same ordered rules for folding a masked
  load and the two orientations of a masked `MAX`. Extracted that three-rule physical materialization block once and
  composed both matchers around it without changing their prefix or suffix rule order.
- Compared milestone-93/current serialized images for a nested masked load and a nonfinite masked-`MAX` selector;
  both were byte-identical. The full Rockchip UOp unit suite also covers the storage-boundary matcher composition.
- Renderer executable size fell from 4,709 to 4,704 lines. From the 10,233-line baseline, 5,529 executable renderer
  lines are gone (54.0%); runtime remains 480 lines. This only deduplicates DPU graph materialization rules and adds no
  CPU numeric semantics.

Verification:

- Old/current `encode_image()` comparison: 2/2 affected materialization programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.17 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 95. Reuse the expanded physical dependency order — complete

- Profiled the slowest unit case, `test_dependent_reduction_range_preserves_vector_output_axis`. After expanding its
  physical recipe, `_lower_uop_program()` already had the exact dependency order, but `RKContext.__init__()` and
  `RKContext.finish()` each independently topologically sorted the same 17.5k-node graph again.
- Passed that ordered node dictionary into `RKContext` and reused it for both context classification and iterative
  lowering. The direct cProfile workload fell from 2,265,951 to 2,059,999 calls and from 1.512 to 1.405 seconds (7.1%);
  `_lower_uop_program()` cumulative time fell from 1.680 to 1.574 seconds (6.3%). A warmed alternating AB/BA benchmark
  of the largest 128x128 case improved from a 0.394-second median to 0.382 seconds (3.0%).
- Scalar, vector, and large dependent-reduction images remained byte-identical to milestone 94. Renderer size remains
  4,704 executable lines, 5,529 lines or 54.0% below baseline; runtime remains 480 lines. This only reuses graph
  metadata and adds no CPU numeric semantics.

Verification:

- Old/current `encode_image()` comparison: 3/3 dependent-reduction programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 96. Deduplicate nonfinite WHERE denominator setup — complete

- A fresh private-symbol audit found no unreferenced renderer component, but the generic typed `WHERE` handler built
  the same inverted selector denominator separately in its NaN and signed-infinity branches. Hoisted that shared DPU
  stage above the correction split; NaN still uses `0/denominator`, while infinity still uses the signed quotient
  correction.
- Compared milestone-95/current images for NaN, positive infinity, and negative infinity in both `WHERE` arms. All six
  images were byte-identical, including scratch coloring and EW ordering.
- Renderer executable size fell from 4,704 to 4,700 lines. From the 10,233-line baseline, 5,533 executable renderer
  lines are gone (54.1%); runtime remains 480 lines. No host computation or CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 6/6 nonfinite `WHERE` programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.02 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 97. Delete impossible BOOL layout guards — complete

- Both BOOL binary lowering and BOOL `WHERE` choose `preferred` from exactly `BOOL_INT16` or `BOOL_MASK`, then checked
  whether it was outside that same two-value set. Removed those impossible branches. BOOL binary lowering also
  rechecked `_coerce_bool()`'s guaranteed postcondition immediately after coercing both operands; removed that dead
  guard as well.
- Removed the one-call `_accurate_add()` forwarding method and routed its dispatch directly through
  `_accurate_add_recipe()` and the context's ordinary memoized `lower()` path.
- Compared milestone-96/current images for composed BOOL binary arithmetic, BOOL `WHERE`, and compensated product ADD;
  all three were byte-identical. Renderer executable size fell from 4,700 to 4,694 lines. From the 10,233-line
  baseline, 5,539 lines are gone (54.1%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 3/3 affected generic programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.04 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 98. Reuse canonical BOOL constant materialization — complete

- Generic BOOL binary lowering manually allocated BOOL constants in either FP16-mask or INT16 form, mutated an
  optional operand list, then asserted that both entries had become physical values. Replaced that block with the
  same `_static(src, preferred)` plus `_coerce_bool()` typed operand comprehension already used by BOOL `WHERE`.
- Compared milestone-97/current images for BOOL operations combining a constant with a dynamic `BOOL_MASK` and with a
  dynamic `BOOL_INT16` load. Both images were byte-identical, proving that canonical constant bits and layouts are
  unchanged.
- Renderer executable size fell from 4,694 to 4,689 lines. From the 10,233-line baseline, 5,544 executable renderer
  lines are gone (54.2%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 2/2 BOOL physical-layout programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.76 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 99. Carry one dependency order through final canonicalization — complete

- Reprofiled `test_dependent_reduction_range_preserves_vector_output_axis`. Its expanded 17.5k-node physical recipe
  was still traversed separately by INT32-neutral detection, math no-op detection, final node sizing, and a synthetic
  STORE/SINK rebuild used only to recreate the already-known `RKOutput` tuple.
- Made the neutral and math passes consume the dependency order already owned by `_lower_uop_program()`, refreshing it
  only when a pass actually returns a different root. Replaced the synthetic STORE/SINK plus `_output_store()` replay
  with direct replacement of `RKOutput`'s value field; `RKContext` never consumed that temporary store or sink.
- The direct cProfile workload fell from 2,059,999 to 1,647,957 calls (20.0%) and from 1.546 to 1.203 seconds (22.2%).
  A warmed alternating AB/BA benchmark of the largest 128x128 case improved from a 0.389-second median to 0.366
  seconds (5.7%).
- Compared milestone-98/current images for scalar/vector/large dependent reductions plus SIN, BOOL `WHERE`, and INT32
  MAX roots; all six were byte-identical. Renderer executable size fell from 4,689 to 4,688 lines. From the 10,233-line
  baseline, 5,545 lines are gone (54.2%); runtime remains 480 lines. This only reuses compiler graph metadata and adds
  no CPU numeric semantics.

Verification:

- Old/current `encode_image()` comparison: 6/6 representative generic programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 100. Unify the IEEE BOOL physical boundary — complete

- `_compare()`'s mixed-type IEEE fallback and `_ieee_bool()` independently lowered the same mask recipe, accepted the
  same `FP16`/`BOOL_MASK` physical layouts, and retyped the result as canonical `BOOL_MASK`. Routed the comparison
  fallback through `_ieee_bool()` so one typed boundary owns that conversion.
- Compared milestone-99/current images for mixed FP16/FP32 `CMPNE` and `CMPEQ`; both were byte-identical.
- Renderer executable size fell from 4,688 to 4,686 lines. From the 10,233-line baseline, 5,547 executable renderer
  lines are gone (54.2%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 2/2 mixed IEEE comparison programs byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.07 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 101. Share canonical masked ternary emission — complete

- BOOL `WHERE` and INT16 `WHERE` separately allocated the same three intermediate values and emitted the same
  `selector*yes`, `1-selector`, inverse-times-no, and final ADD stages. Added one `_masked_where()` physical primitive
  whose arguments retain true ternary arity and whose destination is still allocated after the intermediates.
- Removed BOOL `WHERE`'s redundant reconstruction of the `RKValue` already returned by `_emit()`. The helper derives
  intermediate dtype/layout from the canonical physical `one` value, so `BOOL_MASK`, `BOOL_INT16`, and INT16 retain
  their existing representations.
- Compared milestone-100/current images for root and nested BOOL and INT16 selections; all four were byte-identical,
  including scratch numbering and stage order. Renderer executable size fell from 4,686 to 4,682 lines. From the
  10,233-line baseline, 5,551 lines are gone (54.2%); runtime remains 480 lines. No CPU numeric semantics were added.

Verification:

- Old/current `encode_image()` comparison: 4/4 root/nested typed selections byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.30 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 102. Delete redundant scratch-coloring state — complete

- Reprofiled the dependent reduction after milestone 99. Scratch coloring is now the largest standalone compiler pass,
  but an identity shortcut is impossible in the large case: only 7 of 23,754 EW arguments retain their virtual index.
- Audited the interval allocator instead. Its separate `starts` array duplicated the first-use information already
  represented by `ends == -1` and ordered discovery, so first-touch records now carry their event directly. Its
  `physical_reusable` array was also redundant: only reusable targets enter `active`, and pinned targets are never
  inserted, making the pop-time boolean check invariantly true.
- Deleted both parallel state arrays and their writes/checks. The profiled workload makes 1,062 fewer calls, although
  scratch-coloring wall time remains within noise at about 0.30 seconds, so no speedup is claimed.
- Compared milestone-101/current images for a large unpinned reduction, pinned raw-bit mid-gathers, static selection,
  and host-address gathering; all four were byte-identical. Renderer executable size fell from 4,682 to 4,681 lines.
  From the 10,233-line baseline, 5,552 lines are gone (54.3%); runtime remains 480 lines. No CPU semantics were added.

Verification:

- Old/current `encode_image()` comparison: 4/4 allocator schedule families byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.64 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 103. Merge finite MAX neutral canonicalization — complete

- Final generic canonicalization ran one complete bottom-up walker to replace selected FP16/FP32 negative-infinity
  padding under a root MAX, then a separate context-aware walker to replace INT32_MIN only beneath integer MAX nodes.
  Combined both semantics into one iterative context-aware `_finite_max_neutrals()` pass over the dependency order
  already owned by `_lower_uop_program()`.
- The merged pass retains the exact activation boundaries: finite FP selector replacement only when the program root
  is MAX, and `-2048` replacement only for INT32_MIN reached under integer MAX. Programs requiring neither return the
  original root and node order unchanged.
- Compared milestone-102/current rewrite graphs for FP selector, INT32 neutral, and no-op roots; all three keys were
  identical. The representative FP selector RKImage was also byte-identical.
- Renderer executable size fell from 4,681 to 4,677 lines. From the 10,233-line baseline, 5,556 executable renderer
  lines are gone (54.3%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 canonical graphs and 1/1 affected RKImage identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.33 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 104. Construct compensated FP16 products without dtype replay — complete

- Reprofiled `test_dependent_reduction_range_preserves_vector_output_axis`. Compensated product construction still
  created roughly 7,500 already-proven FP16 multiplications through generic `UOp.alu()`, causing Tinygrad to repeat
  dtype promotion and inference for every physical recipe node.
- Constructed those MUL UOps directly with canonical `dtypes.half` inside `_sub_half()`, `_split_half()`, and
  `_two_product()`. This is limited to the compensated half recipe where both inputs and the result are structurally
  guaranteed FP16; semantic UOps elsewhere retain ordinary dtype inference.
- The complete profiled slow workload fell from 1,646,892 to 1,587,228 calls (3.6%) and from 1.204 to 1.151 seconds
  (4.4%). `_precise_sum_parts()` fell from 0.229 to 0.171 seconds (25.4%). In a warmed alternating benchmark, isolated
  128-term recipe construction improved from a 45.2 ms median to 26.2 ms (42.1%).
- Two-, 64-, and 128-term recipe keys and scalar/vector/large dependent-reduction RKImages matched milestone 103
  exactly. Renderer size remains 4,677 executable lines, 5,556 lines or 54.3% below baseline; runtime remains 480
  lines. The optimization adds no CPU numeric semantics and changes no emitted hardware stage.

Verification:

- Old/current comparison: 3/3 compensated recipe keys and 3/3 dependent-reduction RKImages identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 105. Slot immutable physical ABI records — complete

- Reprofiled the dependent reduction after milestone 104. The hot path creates about 19k immutable `RKArg`,
  `RKScratch`, `RKEWOp`, and related physical records, and generic frozen dataclass initialization alone cost 93 ms.
- Made the renderer's frozen physical ABI and structural definition dataclasses slotted. No renderer/runtime code uses
  dynamic attributes or `__dict__`; serialization remains explicitly field-based through `encode_image()`.
- Frozen dataclass initialization fell from 93.4 to 8.4 ms (91.0%), scratch coloring fell from 306.9 to 279.5 ms
  (8.9%), and total direct cProfile time fell from 1.149 to 1.118 seconds (2.7%). A warmed alternating benchmark of the
  largest case improved from a 0.340-second median to 0.334 seconds (1.8%).
- Scalar/vector/large dependent reductions, host-address gathering, and pinned raw-bit schedules produced image bytes
  identical to milestone 104. Renderer size remains 4,677 executable lines, 5,556 lines or 54.3% below baseline;
  runtime remains 480 lines. This changes Python record storage only and adds no CPU numeric semantics.

Verification:

- Old/current `encode_image()` comparison: 5/5 physical record families byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.02 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 106. Delete unused output sample environments — complete

- `_contiguous_output_samples()` proved affine destination coverage and then built bounded RANGE environment
  dictionaries for its caller. The sole remaining caller only tested whether the return value was `None`; it never
  consumed a sample, so every dictionary, midpoint, endpoint, and Cartesian sample expansion was dead code.
- Replaced the helper with `_contiguous_output()`, a direct boolean proof retaining the same zero-base, range set,
  stride, positive-limit, and total-extent checks. The existing exact static-vector fallback still validates output
  permutations that are not recognized by the affine fast proof.
- Linear, reversed, transposed, duplicate, and scalar index proofs matched milestone 105, and linear/reversed output
  RKImages were byte-identical. Renderer executable size fell from 4,677 to 4,671 lines. From the 10,233-line baseline,
  5,562 lines are gone (54.4%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 5/5 output proofs equivalent and 2/2 output-layout RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.00 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 107. Delete discarded physical return values — complete

- Audited every renderer helper whose calls are used only as statements. `_ew_eq_mask()` returned a newly constructed
  `RKArg` for its known equality slot, but its sole grouped-BOOL caller discarded that object and continued using the
  slot number directly. Changed the helper to an emission-only `None` contract and removed the construction.
- `_widen_exact_int()` similarly created an `RKValue` only to read its `.arg` for one final EW stage, then returned it
  to `finish()`, which discarded it. The output boundary now writes directly to the canonical `self.out` argument and
  returns `None`.
- Grouped OR/AND reduction images and exact FP16-to-INT32 widening images matched milestone 106 byte-for-byte.
  Renderer executable size fell from 4,671 to 4,668 lines. From the 10,233-line baseline, 5,565 lines are gone (54.4%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current `encode_image()` comparison: 2/2 grouped BOOL operators and 1/1 exact INT widening image identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.78 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 108. Reuse one iterative binary flattener — complete

- `_fp32_add_has_product_terms()` maintained a private iterative ADD-tree traversal while `_flatten_binary()` already
  owned the same semantic recursively. Made the shared helper iterative and order-preserving, then reused it in the
  product predicate. This deletes the duplicate traversal and makes all shared ADD/MUL/AND flattening safe for deep
  trees.
- ADD, MUL, and AND leaf ordering matched milestone 107 exactly. Floating-product detection also matched for direct
  FP32 products, cast-FP16 products, non-product trees, and a 4,096-deep ADD tree. Representative sum and nested
  product RKImages were byte-identical.
- Renderer executable size fell from 4,668 to 4,667 lines. From the 10,233-line baseline, 5,566 lines are gone (54.4%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 binary leaf orders, 4/4 product predicates, and 2/2 product RKImages identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.11 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 109. Delete the remaining private FP32 ADD walker — complete

- `_fp32_add_terms()` still recursively traversed the same FP32 ADD tree now owned by `_flatten_binary()`. Replaced its
  local list, closure, recursive descent, and final mutation with one conversion over the shared order-preserving
  flattener.
- Four representative term lists matched milestone 108 exactly, including left- and right-nested trees. Pure-add and
  product-sum storage RKImages were byte-identical, while a 4,096-deep tree now flattens without Python recursion.
- Renderer executable size fell from 4,667 to 4,661 lines. From the 10,233-line baseline, 5,572 lines are gone (54.4%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 4/4 term lists and 2/2 storage RKImages identical; 4,096-deep tree accepted.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.21 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 110. Reuse iterative ADD flattening for FP32 SIN — complete

- FP32 SIN range reduction contained another recursive ADD closure for splitting phase terms and constant residuals.
  Replaced only its traversal with `_flatten_binary()` while retaining the existing per-term conversion and split-half
  residual logic unchanged.
- Left-, right-, and mixed-nested phase expressions produced identical term/residual sequences, and the representative
  shifted-SIN RKImage was byte-identical to milestone 109. Deep additive phase expressions no longer depend on Python
  recursion depth.
- Renderer executable size fell from 4,661 to 4,659 lines. From the 10,233-line baseline, 5,574 lines are gone (54.5%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 phase term/residual sequences and 1/1 shifted-SIN RKImage identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.16 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 111. Cache physical rather than virtual scratch arguments — complete

- Reprofiled `test_dependent_reduction_range_preserves_vector_output_axis`, still the slowest unit at 1.06 seconds in
  the xdist duration census. Its large case creates 8,177 virtual scratch slots that liveness coloring reduces to only
  521 physical slots, but remapping still allocated a zero-offset `RKArg` for every virtual slot.
- Cache one canonical argument per physical slot and select it through the existing virtual-to-physical map. The slot
  coloring, offsets, physical sizes, and encoded program are unchanged; the large case constructs about 94% fewer
  cached remap arguments.
- Scratch coloring/remapping fell from 279.3 to 247.4 ms (11.4%), total direct cProfile time fell from 1.115 to 1.095
  seconds (1.8%), and calls fell from 1,587,225 to 1,555,353. A warmed alternating benchmark of the largest case
  improved from a 0.3295-second median to 0.3248 seconds (1.4%).
- Scalar, vector, and large dependent-reduction RKImages matched milestone 110 byte-for-byte. Renderer remains 4,659
  executable lines, 5,574 lines or 54.5% below baseline; runtime remains 480 lines. No CPU numeric semantics were
  introduced.

Verification:

- Old/current comparison: 3/3 dependent-reduction RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.85 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 112. Delete the accurate-ADD private traversal — complete

- `_accurate_add_recipe()` retained a recursive half-ADD walker because native/tagged ADD nodes must remain opaque.
  Added an optional `plain` boundary to the shared iterative `_flatten_binary()` and used it to stop precisely at ADDs
  carrying a physical recipe tag, then kept the existing FP32-cast expansion over those ordered leaves.
- Ordinary, balanced, and tagged-boundary accurate recipes matched milestone 111 exactly, as did a representative
  nested product-sum RKImage. A 4,096-deep plain ADD tree now flattens without Python recursion.
- Renderer executable size fell from 4,659 to 4,656 lines. From the 10,233-line baseline, 5,577 lines are gone (54.5%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 accurate recipe keys and 1/1 product-sum RKImage identical; 4,096-deep tree accepted.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.66 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 113. Delete the unused static-expression cache contract — complete

- Audited every `_is_static_expr()` call site. Its optional memo dictionary was never supplied, so the cache lookup,
  recursive cache threading, and cache write were unreachable in every renderer execution.
- Removed the unused parameter and reduced the helper to its actual recursive predicate. Static constants, ranges,
  comparisons, selectors, loads, and dynamic arithmetic produced the same seven predicate results as milestone 112;
  representative static-selector and dynamic-load RKImages were byte-identical.
- Renderer executable size fell from 4,656 to 4,653 lines. From the 10,233-line baseline, 5,580 lines are gone (54.5%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 7/7 static predicates and 2/2 RKImages identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.14 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 114. Collapse repeated WHERE recipe dispatch — complete

- The generic `_where()` handler repeated the same two-line “try matcher, lower returned UOp recipe” control flow for
  absolute value, ordered MIN/MAX selection, and finite-threshold selection. Replaced the three copies with one fixed
  ordered matcher loop; recipe priority and physical lowering remain unchanged.
- Representative ordered-minimum, absolute-value, and threshold-selection programs produced RKImages byte-identical
  to milestone 113.
- Renderer executable size fell from 4,653 to 4,649 lines. From the 10,233-line baseline, 5,584 lines are gone (54.6%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 WHERE recipe RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.83 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 115. Share static vector-evaluation setup — complete

- `_static_values()` and `_static_int_vectors()` independently collected output RANGE dependencies, rejected foreign
  ranges, enumerated bounded environments, built NumPy lane vectors, and initialized the expression cache. Moved that
  identical address/static-materialization setup into `_static_vector_env()`.
- Scalar static values, three simultaneously evaluated integer rows, foreign-range rejection, and a representative
  multi-axis static-selection RKImage matched milestone 114 exactly. This helper evaluates compile-time indices and
  constants only; it does not execute tensor numeric semantics on the host.
- Renderer executable size fell from 4,649 to 4,646 lines. From the 10,233-line baseline, 5,587 lines are gone (54.6%);
  runtime remains 480 lines.

Verification:

- Old/current comparison: scalar/vector values, rejection behavior, and 1/1 RKImage identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.68 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 116. Centralize comparison arity validation — complete

- FP16 equality and less-than handlers repeated binary arity plus op/dtype checks after `_compare()` had already
  selected their exact semantic family. Moved the true two-source comparison rule to `_compare()` once and deleted the
  unreachable private-handler checks. Bool, integer, FP16, and IEEE fallback comparisons now share the same arity
  boundary.
- FP16 `CMPLT`, `CMPNE`, and `CMPEQ` programs produced RKImages byte-identical to milestone 115.
- Renderer executable size fell from 4,646 to 4,643 lines. From the 10,233-line baseline, 5,590 lines are gone (54.6%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 FP16 comparison RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.20 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 117. Share bounded physical-image working lists — complete

- Four bounded integer/address recipes independently created the same scratch-size list, slot allocator closure,
  gather list, and EW-stage list. Added `_physical_lists(minimum)` as one physical construction helper; candidate and
  predicate-coordinate recipes retain their exact 64-byte minimum, while dynamic gather and bounds-mask recipes keep
  exact requested sizes.
- Candidate INT32 selection, dynamic typed loading, bounded FP16 predicate coordinates, and normalized INT32 bounds
  programs produced RKImages byte-identical to milestone 116.
- Renderer executable size fell from 4,643 to 4,635 lines. From the 10,233-line baseline, 5,598 lines are gone (54.7%);
  runtime remains 480 lines. The helper owns allocation bookkeeping only; no CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 4/4 affected physical-builder RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.69 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 118. Bypass MAX-only operand normalization in ordinary ALU stages — complete

- Reprofiled `test_dependent_reduction_range_preserves_vector_output_axis`. Its roughly 15,900 ordinary ALU stages
  each allocated and called a nested operand function whose only special behavior applies to FP16 MAX with negative
  infinity. ADD/MUL/SUB still paid two calls and both MAX-only condition chains.
- Kept the existing finite-neutral operand function only inside FP16 MAX and call the typed `_operand()` boundary
  directly for every other ALU UOp. Negative-infinity constants and statically masked loads retain their exact
  normalization.
- Complete cProfile time fell from 1.115 to 1.078 seconds (3.3%), `_alu()` cumulative time fell from 325 to 304 ms
  (6.5%), and calls fell from 1,555,350 to 1,523,475. A warmed alternating benchmark of the largest case improved from
  a 0.3183-second median to 0.3097 seconds (2.7%).
- Scalar/vector/large dependent reductions and both negative-infinity MAX forms produced RKImages byte-identical to
  milestone 117. This measured fast path costs three executable lines, temporarily taking the renderer from 4,635 to
  4,638 lines; runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 5/5 reduction/MAX RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.89 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 119. Make finite-MAX materialization a normal operand boundary — complete

- Rewrote milestone 118's conditional nested function as `_max_operand()`. `_alu()` now selects one bound operand
  implementation and invokes it at the true binary arity, keeping ordinary arithmetic on the direct typed path while
  making the exceptional MAX boundary independently readable and reusable.
- The refactor removes two of milestone 118's three added executable lines. Direct cProfile remained within run noise
  (1.078 versus 1.082 seconds, identical 1,523,475 calls), and scalar/vector/large reductions plus both finite-MAX
  normalization forms remained byte-identical.
- Renderer executable size fell from 4,638 to 4,636 lines. From the 10,233-line baseline, 5,597 lines are gone (54.7%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 5/5 reduction/MAX RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.74 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 120. Merge identical physical CAST branches — complete

- The generic CAST handler separately dispatched INT32-backed values to the same narrowing recipe for half and float
  outputs. It also duplicated the identical BOOL_INT16 selection recipe for half and float outputs. Merged each pair
  around its canonical source layout while retaining the prior source/destination dtype guards.
- INT32→half, INT32→float, BOOL_INT16→half, and BOOL_INT16→float programs produced RKImages byte-identical to milestone
  119. No additional conversion pair is accepted.
- Renderer executable size fell from 4,636 to 4,631 lines. From the 10,233-line baseline, 5,602 lines are gone (54.7%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 4/4 merged CAST RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 5.05 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 121. Consolidate CAST ABI relabeling and BOOL recipes — complete

- Merged the remaining FP16-backed identity conversions for semantic half/float and the native-INT16 identity
  conversions for semantic INT16/bounded INT, retaining every prior source-layout and dtype guard.
- BOOL_INT16 conversion to half, float, or INT_FP16 now builds its shared 0/1 selection recipe once; only the INT
  destination applies the existing physical-layout relabel afterward.
- Half→float, embedded-int→half, BOOL_INT16→half, packed-bool→INT16, and packed-bool→INT programs produced RKImages
  byte-identical to milestone 120. Renderer executable size fell from 4,631 to 4,628 lines. From the 10,233-line
  baseline, 5,605 lines are gone (54.8%); runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 5/5 identity/BOOL CAST RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.76 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 122. Delete single-use CAST recipe temporaries — complete

- FP32→half and exact-int→half preconversion assigned local `recipe` values only to pass them immediately back to
  `RKContext.lower()`. Removed those dead temporaries and materialized the corresponding UOp recipes at their single
  use, while retaining the direct FP32 LOAD boundary.
- Direct FP32 LOAD→half, composed FP32 expression→half, and bounded semantic INT→half programs produced RKImages
  byte-identical to milestone 121.
- Renderer executable size fell from 4,628 to 4,624 lines. From the 10,233-line baseline, 5,609 lines are gone (54.8%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 preconverted CAST RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.65 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 123. Delete remaining one-use generic recipe locals — complete

- Continued the local-use census through generic CAST and CMOD lowering. Removed recipe variables used exactly once
  for half→uchar truncation, half→bool nonzero conversion, BOOL_INT16 selection, FP16-backed INT conversion, and
  INT_FP16 remainder. Each UOp recipe is now constructed directly at its sole `lower()` boundary.
- Half→uchar, half→bool, packed-bool→half, half→INT_FP16, and INT_FP16 CMOD programs produced RKImages byte-identical
  to milestone 122.
- Renderer executable size fell from 4,624 to 4,618 lines. From the 10,233-line baseline, 5,615 lines are gone (54.9%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 5/5 single-use recipe RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.77 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 124. Delete final one-read INT_FP16 locals — complete

- Removed the remaining one-read UOp/value locals in bounded integer comparison, FP16-backed integer CAST, and
  INT_FP16 CMOD. Each recipe is lowered exactly once and its physical argument is consumed directly where required.
- Bounded integer comparison, half→INT_FP16, and INT_FP16 remainder programs produced RKImages byte-identical to
  milestone 123.
- Renderer executable size fell from 4,618 to 4,615 lines. From the 10,233-line baseline, 5,618 lines are gone (54.9%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 3/3 affected RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.74 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 125. Give all comparison UOps one semantic dispatcher — complete

- `lower()` separately intercepted half comparisons, direct boolean equality, IEEE-derived equality, and then remaining
  comparisons before they converged on `_compare()`. Routed every `CMPLT`, `CMPNE`, and `CMPEQ` through `_compare()`
  first; that handler already owns half, bool, bounded/wide integer, INT16, and IEEE fallback semantics. Boolean
  `AND/OR/XOR` retain their direct and derived-mask paths.
- Half, bool, INT32, mixed half/float comparisons and direct boolean AND produced 11 RKImages byte-identical to
  milestone 124.
- Renderer executable size fell from 4,615 to 4,614 lines. From the 10,233-line baseline, 5,619 lines are gone (54.9%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 11/11 comparison/logical RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.76 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 126. Give boolean ALU UOps one semantic dispatcher — complete

- After comparisons moved to `_compare()`, three overlapping `AND/OR/XOR` branches remained. Collapsed them into one
  boolean-ALU block: canonical bool operands lower directly, noncanonical operands may use the existing IEEE mask
  recipe, and unmatched forms retain the ordinary `_bool_binary()` rejection/handling path.
- Direct bool and noncanonical half-backed `AND`, `OR`, and `XOR` programs produced the same six accepted/rejected
  results and byte-identical RKImages as milestone 125.
- Renderer executable size fell from 4,614 to 4,613 lines. From the 10,233-line baseline, 5,620 lines are gone (54.9%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 6/6 boolean dispatch results identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.90 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 127. Unify scalar and vector static expression evaluation — complete

- Replaced the separate scalar and NumPy-vector static opcode evaluators with one typed evaluator. Scalar mode retains
  short-circuit `WHERE` and tinygrad integer division/remainder helpers; vector mode retains NumPy broadcasting,
  overflow casting, masked zero-divisor behavior, and eager `WHERE` materialization.
- The evaluator remains limited to compile-time constants and static index/address construction. It does not execute
  runtime tensor arithmetic on the CPU, and the generated hardware programs are unchanged.
- Renderer executable size fell from 4,613 to 4,598 lines. From the 10,233-line baseline, 5,635 lines are gone (55.1%);
  runtime remains 480 lines.

Verification:

- Old/current evaluator comparison: 38/38 scalar, vector, overflow, division, and short-circuit cases identical.
- Old/current program comparison: 5/5 representative static gather, selector, integer, and predicate RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.77 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 128. Restore generic mapped execution for FP32 local reductions — complete

- Profiled the serial Rockchip UOp suite. `test_dependent_reduction_range_preserves_vector_output_axis` was the
  slowest case at 0.66 seconds because the mapped `RANGE`/local `LOAD`/`STORE` executor searched only the final value
  DAG; current stateful UOps keep the accumulator update as a sibling program node, so it fell through to literal
  expansion of thousands of compensated EW stages.
- Made the existing structural reducer discover reduction ranges and local stores from the complete UOp program.
  The fast path is limited to FP32 local accumulators; FP16 ADD/MAX/MUL loops retain their literal generic behavior.
  This is structural UOp execution, not tensor-operation recognition and not host numeric evaluation.
- Cached the two canonical immutable scratch specifications and bypassed special-ALU dispatch for ordinary physical
  FP16 arithmetic. The three profiled shapes now lower in a 0.0109-second median instead of 0.6338 seconds (58.1x),
  with 66/66/129 EW stages instead of 4,012/4,012/7,918. The former slowest test is now 0.03 seconds, and the serial
  91-test suite fell from 2.60 to 1.82 seconds.
- Renderer executable size rose from 4,598 to 4,601 lines for the repaired structural boundary. From the 10,233-line
  baseline, 5,632 lines remain deleted (55.0%); runtime remains 480 lines.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.66 seconds.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0 --durations=15`: 91 passed in 1.82 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 129. Delete the duplicate static-local arithmetic interpreter — complete

- `_static_local_load_offsets()` carried a second scalar/NumPy implementation of CAST, ADD, SUB, MUL, MAX,
  comparisons, boolean ALU, integer division/remainder, NEG, and WHERE. Extended the shared static UOp evaluator with
  structural `AFTER`, unary `NEG`, and an optional local-LOAD resolver, then routed local address materialization
  through it.
- Extracted one shared typed binary evaluator for both ordinary static indices and bounded local address programs.
  Host execution remains restricted to compile-time index/address calculation; runtime tensor numeric semantics still
  execute on the DPU.
- Renderer executable size fell from 4,601 to 4,575 lines. From the 10,233-line baseline, 5,658 lines are gone (55.3%);
  runtime remains 480 lines.

Verification:

- Old/current static evaluator comparison: 30/30 scalar/vector opcode cases identical.
- Old/current nested local-address program: RKImage byte-identical, including `(0, 0, 0, 0)` gathered offsets.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.55 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 130. Unify affine static-index analysis — complete

- Replaced the separate affine and divided-affine recursive analyzers with one linear static-address analyzer. A typed
  wrapper selects whether ordinary RANGE atoms or `RANGE//constant` atoms are admitted; CONST/ADD/SUB/MUL coefficient
  propagation and cancellation now have one implementation.
- This changes no runtime address behavior and introduces no host numeric evaluation. Eight ordinary, divided,
  cancelled, casted, and deliberately non-affine expressions matched the old analyzers exactly; the compact
  million-lane divided gather plan was also identical.
- Renderer executable size fell from 4,575 to 4,564 lines. From the 10,233-line baseline, 5,669 lines are gone (55.4%);
  runtime remains 480 lines.

Verification:

- Old/current analyzer comparison: 8/8 expressions and 1/1 divided gather plan identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.66 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 131. Give physical RKImages one argument-remap ABI — complete

- Mapped reductions, appended post-reduction programs, multi-local composition, and scalar-extrema execution each
  carried private copies of the same relocation logic for EW operands, gather endpoints, and host-address records.
  Added one `_map_image_args()` physical-ABI primitive and deleted those four handwritten remappers.
- Scratch specifications, constants, scheduling points, and semantic UOps are untouched. Old/current mapped FP32
  reduction, generic post-op, dependent scalar extrema, and two-local composition programs produced four byte-identical
  serialized RKImages.
- Renderer executable size fell from 4,564 to 4,543 lines. From the 10,233-line baseline, 5,690 lines are gone (55.6%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current comparison: 4/4 relocated RKImages byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.67 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 132. Delete the scratch allocator's private relocation pass — complete

- `_reuse_linear_scratch()` still rebuilt gathers, EW stages, and host-address records with a private copy of the
  relocation logic introduced in milestone 131. Routed its virtual-to-physical slot coloring through
  `_map_image_args()` and deleted the duplicate gather/host/EW reconstruction.
- Added the one required distinction to the shared ABI: compile-time value gathers retain their ignored source field
  during scratch coloring. The remaining explicit 64-term non-affine MUL/ADD program produced a byte-identical image.
- Its alternating renderer-only benchmark remained stable at 0.096384 seconds versus 0.096578 seconds before the
  refactor. This case deliberately remains literal ordinary UOp execution rather than a recovered matmul dialect.
- Renderer executable size fell from 4,543 to 4,535 lines. From the 10,233-line baseline, 5,698 lines are gone (55.7%);
  runtime remains 480 lines. No CPU numeric semantics were introduced.

Verification:

- Old/current non-affine reduction RKImage: byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.61 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 133. Give every runtime EW tile one offset-argument boundary — complete

- The runtime's scratch-INT16, spatial, INT16→INT32, native-integer, comparison, and ordinary FP16 execution branches
  each rebuilt destination/lhs/rhs `RKArg` offsets independently. Added `_offset_ew_args()` and routed all six tiling
  paths through the same physical argument calculation.
- Twelve combinations of destination and shared input offsets matched the removed constructors exactly. This changes
  neither DPU stage flags nor host behavior and introduces no CPU numeric semantics.
- Runtime executable size fell from 480 to 470 lines. Renderer remains 4,535 executable lines, 5,698 lines below its
  10,233-line baseline (55.7%).

Verification:

- Offset-argument comparison: 12/12 cases identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.53 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 134. Give runtime EW execution one PC-body flush transition — complete

- `_run_ew_ops()` repeated the same accumulated-PC submission and list reset across barrier, FP32 output, native
  precision changes, INT32 conversion, compare, and terminal paths. Replaced those copies with one local `flush()`
  transition while retaining every existing precision-state update and reset point.
- Fake-device replay of mixed ordinary/barrier/compare/INT16, terminal FP32, and INT32-conversion programs produced
  three identical submission-body/reset sequences before and after the rewrite.
- Runtime executable size fell from 470 to 464 lines. Renderer remains 4,535 executable lines, 5,698 lines below its
  10,233-line baseline (55.7%). No CPU numeric semantics were introduced.

Verification:

- Old/current runtime execution-sequence comparison: 3/3 identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.55 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 135. Delete duplicate PC-chain size calculation — complete

- `_submit_pcchain()` already walks every body to build aligned command offsets and the total qword count, but then
  `_pcchain_sizes()` independently walked the same lengths to compute an asserted-equivalent allocation size. Deleted
  the second pass and derive command and task-buffer sizes directly from the authoritative offset walk.
- Five single/multi-body, odd/even, and maximum-sized PC-chain cases produced identical command and descriptor sizes.
  Validation retains the same body-count and 16-bit register-amount boundary.
- Runtime executable size fell from 464 to 460 lines. Renderer remains 4,535 executable lines, 5,698 lines below its
  10,233-line baseline (55.7%). No CPU numeric semantics were introduced.

Verification:

- Old/current PC-chain size comparison: 5/5 identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.41 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 136. Unify RKImage gather decoding — complete

- Affine, explicit-offset, embedded-value, and partial gather records each rebuilt the same `RKGather` ABI fields in
  separate decoder branches. Decode only the kind-specific payload now, then construct the physical gather once.
- All four gather kinds, including 8/16/32-bit embedded values, preserved every decoded field and serialized byte.
  This is physical image handling only; no CPU numeric semantics were introduced.
- Renderer executable size fell from 4,535 to 4,526 lines. From the 10,233-line baseline, 5,707 lines are gone (55.8%);
  runtime remains 460 lines.

Verification:

- Old/current gather decode comparison: 6/6 payload variants field-identical; 322/322 serialized bytes identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.88 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 137. Collapse and accelerate the physical EW record — complete

- Profiling the remaining slowest unit case showed that its 3,951 ordinary MUL/ADD stages spent 69 ms under cProfile
  in `decode_image()`. The 40-byte EW record was decoded with three struct formats and encoded with three packs even
  though it is one contiguous physical ABI record. Replaced the fragments with one authoritative struct.
- Decode now reuses the two canonical `RKBufferKind` instances rather than performing 11,853 enum constructions for
  that image. The unsigned-byte flags field also made its `op_flags & ~0xff` validation dead code, so it was deleted.
- Controlled alternating measurements on the slowest 163,352-byte image improved median decode from 0.031153 to
  0.024846 seconds (20.2%) and encode from 0.007261 to 0.005462 seconds (24.8%). The complete image remained
  byte-identical. The math-recipe guard now scans one shared UOp DAG instead of separately traversing every term.
- Renderer executable size fell from 4,526 to 4,522 lines. From the 10,233-line baseline, 5,711 lines are gone (55.8%);
  runtime remains 460 lines. The non-affine case remains literal UOp execution; no graph dialect or CPU numeric
  semantics were introduced.

Verification:

- Old/current slowest-case RKImage: 3,951 stages and 163,352/163,352 serialized bytes identical.
- Old/current EW record matrix: 6/6 flag, buffer-kind, and addend variants; 338/338 bytes identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed in 4.94 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 138. Delete the overlapping direct dynamic-index catalog entry — complete

- The 48-line direct dynamic typed-LOAD recognizer and the general multi-index materializer duplicated the same native
  raw-byte selection for one positive or negative-normalized INT32 index. Extended the general materializer with its
  only missing capability—one external raw bool gate—then deleted the direct recognizer and both dispatch references.
- Positive FP16, externally gated FP16, and direct INT32 programs produce byte-identical images to the removed path.
  Negative-normalized input now uses the general executor's existing alternate-coordinate representation, which shares
  one positive candidate plan between `i` and `i-extent` rather than constructing two operation-specific plan sets.
- Added a regression proving that the composed external bool gate remains `NATIVE`, enters as a raw one-byte gather,
  and introduces no host-address or CPU numeric execution.
- Renderer executable size fell from 4,522 to 4,492 lines. From the 10,233-line baseline, 5,741 lines are gone (56.1%);
  runtime remains 460 lines.

Verification:

- Old/current positive FP16 image: 3,917/3,917 bytes identical.
- Old/current externally gated FP16 image: 4,058/4,058 bytes identical.
- Old/current direct INT32 image: 9,103/9,103 bytes identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 92 passed in 5.04 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 139. Delete root INT32-bounds graph recovery — complete

- Disconnected the 54-line recognizer that recovered positive and negative-normalized INT32 bounds predicates as a
  special native byte-mask image. The generic typed executor now accepts the same `LOAD`, `CMPLT`, `CMPNE`, `WHERE`,
  and `AND` UOps directly, including conjunctions over independent index tensors.
- Deleted the recognizer and its pre-generic dispatch. The generic path retains exact INT32 byte-plane comparisons,
  native DPU boolean arithmetic, and raw one-byte output materialization; no host address or CPU numeric execution is
  involved.
- Added a two-axis negative-normalization regression. Compared current images with milestone 138 after disconnecting
  only the obsolete recognizer: positive/normalized one-axis and two-axis programs were byte-identical in all four
  cases, proving this exposes an already-existing generic path rather than introducing a replacement dialect.
- Renderer executable size fell from 4,492 to 4,438 lines. From the 10,233-line baseline, 5,795 lines are gone (56.6%);
  runtime remains 460 lines.

Verification:

- Milestone-138 generic/current comparison: 4/4 images byte-identical (110, 176, 219, and 351 ordinary EW stages).
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 93 passed in 4.69 seconds.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0 --durations=10`: 93 passed in 1.95 seconds;
  the literal non-affine UOp program remains the slowest at 0.21 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 140. Give runtime typed EW tiling one stage builder — complete

- Scratch INT16 execution, INT16→INT32 conversion, native INT16/INT32 execution, and compare submission each repeated
  the same count clipping, destination/source byte-offset calculation, stage emission, and relocation. Routed all four
  paths through one `_ew_stages()` physical builder while preserving their precision and state flags.
- Old/current fake-device replay covered scratch INT16, spatial stateful tiling, INT16→INT32, native INT32,
  compare/standalone, and barrier programs. All six produced identical command bodies, PC-chain groupings, standalone
  submissions, and reset events. This changes only physical submission construction and adds no host numeric work.
- Runtime executable size fell from 460 to 454 lines. Renderer remains 4,438 executable lines, 5,795 lines below its
  10,233-line baseline (56.6%).
- The full hardware census remains pending: both the normal reset ioctl and the local `(RESET, 6, RESET)` recovery
  sequence completed, but `~/rk3588/examples/elementwise.py` still timed out with `Errno 110` on submit.

Verification:

- Old/current physical execution replay: 6/6 precision/scheduling families identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 93 passed in 4.47 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 141. Keep static FP32 coordinates outside compensated-stage tagging — complete

- Recovered the NPU after the earlier timeout and completed the first full hardware census on this branch:
  423 passed, 12 failed, 12 skipped, and 152 subtests passed in 19:01. The failures were four remaining semantic cases,
  two average-pool subtests, and six interpolation methods; none was an NPU timeout.
- Removed `RKContext.static_nodes`, whose construction-time catalog became stale when generic storage rewrites created
  new static UOps. Static values are now classified directly from their UOp dependency graph, and the classifier uses
  Tinygrad's iterative topological walk instead of Python recursion.
- Extended the generic FP32 storage boundary through semantic `WHERE` and `TRUNC` UOps. Tightened compensated-stage
  tagging to physical FP16 `ADD` only, so static FP32 coordinate graphs remain exact materializations instead of being
  retagged and lowered as lossy dynamic half arithmetic.
- The two failing average-pool padding subtests and all six interpolation methods now pass on hardware. Focused timing
  improved one-dimensional interpolation from 3.64 seconds while failing to 0.49 seconds while passing; the complete
  seven-method hardware gate passes in 27.08 seconds. No tensor-operation recognizer or host numeric execution was
  introduced.
- The deleted static catalog offsets the two generic storage handlers: renderer executable size remains 4,438 lines,
  5,795 lines below its 10,233-line baseline (56.6%); runtime remains 454 lines.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest` over the repaired average-pool method and
  all six interpolation methods with `-q -n0`: 7 passed, 3 subtests passed in 27.08 seconds.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 94 passed in 6.21 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: pass.
- `git diff --check`: pass.

### 142. Share static UOp materialization and bound hardware lanes — complete

- Added the missing integer bound semantic for `SPECIAL`: like `RANGE`, a hardware lane with constant extent `N` is
  exactly bounded by `[0, N-1]`. This lets ordinary integer `ADD/MUL/MAX` UOps choose their physical layout without a
  pool or unpool recognizer. Max-pool return indices and max-unpool now both pass on hardware.
- Profiled the newly passing max-unpool case, which became the observed slowest test at 19.85 seconds. cProfile found
  20.70 of 37.65 seconds in 152 repeated static-vector materializations, including 5.1 million scalar NumPy `.item()`
  conversions and repeated reconstruction of the same 60,000-lane output environment.
- `RKStaticIndexEvaluator` now owns typed static values as well as gather offsets. One context shares its range vectors,
  output permutation, and expression cache across all static UOps; common INT, INT16, and FP16 encodings stay in NumPy
  arrays instead of crossing Python once per lane. Max-unpool fell from 19.85 to 9.30 seconds (53.1%) and remains exact.
- Renderer executable size is 4,445 lines, still 5,788 lines below its 10,233-line baseline (56.6%); runtime remains
  454 lines. The seven-line increase is the shared typed materialization API; it replaces repeated work, not UOp or
  tensor-operation specialization, and performs no host numeric tensor semantics.

Verification:

- Max-pool return indices: hardware pass in 2.90 seconds.
- Max-unpool: hardware pass in 9.30 seconds, down from 19.85 seconds.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 95 passed in 5.89 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: pass.
- `git diff --check`: pass.

### 143. Make large precision lowering linear and restore argmax fallback — complete

- The new slowest failure, cross-entropy reduction, spent more than 90 seconds at 100% CPU before reaching hardware.
  An interrupt trace found repeated `UOp.toposort()` calls while the large-program recursion guard eagerly lowered every
  intermediate `ADD` and independently rebuilt overlapping compensated recipes.
- Static classification is now an incremental `UOp.topovisit()` property of `RKContext`, so generated recipes remain
  classifiable without the stale catalog removed in milestone 141 and each dependency is classified once. For large
  graphs, the iterative pre-materializer defers only top-level FP16 precision boundaries and their consumers, then each
  compensated sum is expanded once. The same cross-entropy case now reaches its numeric assertion in 6.22 seconds.
- Hardware probing found the scalar-extrema fast path's exact-byte equality is reliable through 64 lanes and for aligned
  larger vectors, but returns no selected lane for non-32-aligned sizes such as 100, 129, and 200. Those physical shapes
  now use the already-existing literal local-UOp executor. The complete upstream argmax method passes in 1.98 seconds,
  including its formerly incorrect 200-element reduction; no argmax graph dialect or CPU arithmetic was added.
- Cross-entropy `sum` is the one remaining focused failure: 4.145 versus 4.11 at a 0.0255 combined tolerance. Its
  `none` output passes elementwise tolerance and `mean` passes; this is now a numeric accuracy issue, not a compiler or
  submission stall.
- Renderer executable size is 4,459 lines, 5,774 lines below its 10,233-line baseline (56.4%); runtime remains 454.

Verification:

- Argmax hardware method: 1 passed in 1.98 seconds.
- Cross-entropy reduction: reaches its sole numeric failure in 6.22 seconds, down from more than 90 seconds.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 95 passed in 4.65 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: pass.
- `git diff --check`: pass.

### 144. Delete bounded INT32 lookup recovery and vectorize static address planning — complete

- Deleted the bounded-INT32-lookup tensor-pattern recognizer, its private selection image, two static-vector helpers,
  and the pre-generic dispatch. A literal regression now proves the same `LOAD`, bounds predicates, `WHERE`, integer
  arithmetic, and `STORE` program lowers through the ordinary typed UOp executor as `NATIVE`, with no host gather.
- The removed catalog accounted for 64 executable renderer lines. The complete milestone is a net 59-line reduction
  after adding the regression and sharing one `RKStaticIndexEvaluator` across all candidate gather plans.
- Profiled the slowest observed ported test, `test_slice_fancy_indexing_dim_inject_none`. Its first subcase repeatedly
  rebuilt the same output RANGE environment for 303 plans and converted millions of compile-time address coordinates
  through Python generators. The shared evaluator and NumPy-backed tuple materialization retain exactly the same raw
  gather/EW execution while reducing the full method from 26.70 to 18.01 seconds (32.5%).
- The affected hardware gate passes all 13 gather, masked-select, and fancy-index tests in 62.61 seconds. These changes
  perform only compile-time address and comparison-coordinate materialization on the host; all runtime index comparison,
  selection, and tensor values remain native NPU operations.
- Renderer executable size fell from 4,459 to 4,400 lines, 5,833 lines below its 10,233-line baseline (57.0%); runtime
  remains 454 lines.

Verification:

- Affected hardware gate: 13 passed in 62.61 seconds with `-n0`.
- Slowest affected method: 1 passed in 18.01 seconds, down from 26.70 seconds.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 96 passed in 4.46 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 145. Delete scalar extrema recovery and profile the complete census bottleneck — complete

- Completed the timed 445-test `test/backend/test_rockchip.py` census (not `test_rockchip2.py`): 428 passed, 5 failed,
  12 skipped, and 154 subtests passed in 17:44. One failure was the deliberately disconnected bounded-coordinate probe,
  which proved the large `nonzero` `IF/STORE` form is not yet generic and was restored. Cross-entropy `sum` remains the
  known precision failure. Scatter-`inf`, causal-attention NaNs, and cumulative `EINVAL` appeared only in the long shared
  device run and remain pending clean replay after NPU recovery.
- Deleted the 141-line dependent scalar-local extrema recognizer and its dispatch. The literal RANGE/local
  `LOAD`/`STORE` unit program now selects canonical INT32 for an unknown integer range and executes as 203 ordinary native
  stages instead of asserting the removed fast-path shape. The complete upstream argmax method passes in 2.00 seconds;
  softmax+argmax also passes without the recognizer in 136.14 seconds.
- The census identified softmax+argmax as the actual slowest method at 143.18 seconds. Both axes cost roughly the same:
  64.71 and 66.49 seconds when measured independently. cProfile of one axis recorded 247 million calls, 122.5 seconds
  in rendering, 93.7 seconds repeatedly expanding 2,925 math recipes, 403,743 EW stages, 26,325 physical barriers, and
  29,630 PC-chain submissions under instrumentation.
- Large-root math deferral now keys off the post-structural root actually given to `RKContext`, rather than the smaller
  pre-unroll storage graph. This reduces one strict axis from 64.71 to 49.30 seconds (23.8%) while preserving the literal
  UOp execution and all native arithmetic.
- Tested coalescing stateful FLOOR/FDIV barriers after four focused division/transcendental methods passed. A single
  unbounded 403k-task chain exceeded the driver limit with `EINVAL`, so that experiment and its runtime cap were reverted;
  no unverified batching change remains. The oversized submit left health checks returning `EINVAL` despite reset flags
  6 and 1→6→1, so further hardware replay requires an external platform reset or reboot.
- Renderer executable size fell from 4,400 to 4,262 lines, 5,971 lines below its 10,233-line baseline (58.4%); runtime
  remains 454 lines.

Verification:

- Generic argmax and softmax+argmax hardware gate: 2 passed in 141.58 seconds.
- Large-root guard, one softmax+argmax axis: passed in 49.30 seconds, down from 64.71 seconds.
- Stateful-chain semantic probe before the oversized submission: sqrt, exp2, log2, and division-rounding all passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 96 passed in 4.51 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 146. Give scalar local accumulators one structural parser — complete

- Removed the separate 31-line single-buffer local-accumulator parser and its output-range heuristic. One or many
  scalar local buffers now use `_static_local_defs` plus the same dependency expander for generic ADD/MAX/MUL RANGE
  execution.
- Hand-authored UOps do not always attach their reduction RANGE through `AFTER`; the shared parser now falls back to
  the UOp's explicit `AxisType.REDUCE` classification. This is a structural UOp rule and introduces no tensor-operation
  recovery or host numeric execution.
- Compared milestone-145/current serialized images for literal scalar-local ADD, MAX, and MUL programs. All three are
  byte-identical at 303 bytes, including gather, scratch, EW-stage, and output layout decisions.
- Renderer executable size fell from 4,262 to 4,234 lines, 5,999 lines below its 10,233-line baseline (58.6%); runtime
  remains 454 lines.
- NPU replay remains unavailable: the mandatory health check passes through 4,096 lanes but times out at 131,072, and
  a later driver interaction produced an RKNPU kernel trace. No further `/dev/dri` access was attempted in this
  milestone; verification is deliberately limited to host-independent image construction and static checks.

Verification:

- Old/current scalar-local ADD/MAX/MUL images: 3/3 byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 96 passed in 4.47 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 147. Delete grouped boolean-reduction recovery — complete

- Deleted the grouped `all`/`any` graph recognizer, its four private image builders, its block-reduction helper, and a
  now-dead FP16 equality-stage helper. Boolean reductions now use the same literal local-UOp executor as scalar
  ADD/MAX/MUL reductions; no tensor-operation name or graph dialect remains in their dispatch.
- Extended the structural executor with the actual missing semantics: scalar AND/OR accumulators and a strictly proven
  identity-indexed LOCAL bridge. AFTER ordering dependencies are excluded from semantic local-load discovery, and the
  discovery is memoized so large math DAGs remain linear to inspect.
- Captured the exact Rockchip codegen UOps for `Tensor.ones(1<<15).bool().all()` through a NULL device, including its
  SPECIAL/RANGE/LOCAL/BARRIER/IF structure. Both reduction kernels lower through the generic path as NATIVE, with no
  host gathers or host arithmetic. A smaller literal two-level LOCAL bridge is now a permanent unit regression.
- Profiling the initially generic result found direct `CMPNE(half LOAD, 0)` repeatedly entering the full raw-byte IEEE
  equality recipe. The CMPNE UOp handler now uses its exact ABS/nonzero physical recipe: NaN, infinity, signed zero, and
  ordinary FP16 values retain the required bool semantics. For the captured 32K-element program, compile time fell from
  0.429 to 0.101 seconds and the two physical images fell from 5,305 to 673 EW stages. The deleted catalog produced
  1,804 stages, so the generic result is also 62.7% smaller by EW-stage count.
- Renderer executable size fell from 4,234 to 4,136 lines, 6,097 lines below its 10,233-line baseline (59.6%); runtime
  remains 454 lines.
- Hardware replay and the 455-test census remain deferred because the NPU is still in the post-crash state documented
  in milestone 146. This milestone did not open `/dev/dri`; its exact codegen capture used DEV=NULL.

Verification:

- Exact Rockchip large-`all` UOps: 2/2 generic NATIVE images, 673 total EW stages, 0.101-second host lowering.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 97 passed in 4.54 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 148. Delete obsolete isclose graph recovery — complete

- Deleted `_isclose_match` and its post-hoc comparison-mask correction. Tinygrad's current `isclose` UOps already carry
  exact equality, finite tolerance, infinity equality, and optional NaN equality explicitly, so recovering the original
  tensor operation was both architecturally wrong and behaviorally inactive.
- Captured current Rockchip codegen for default tolerance, `equal_nan=True`, and `rtol=0.01` through DEV=NULL. Before
  deletion, forcing the matcher off produced byte-identical images for all three variants; after deletion they retain
  the same 494/496/494 EW-stage counts and 21,899/21,987/21,899-byte serializations.
- The required multi-scalar local executor was deliberately retained. Its previous deletion probe expanded
  `std_mean` into 13,125 loop iterations, so it is structural bounded execution rather than dead catalog code.
- No CPU arithmetic or host value inspection was added. Comparison semantics remain owned by CMPNE/CMPLT/AND/OR UOps
  and their canonical physical boolean recipes.
- Renderer executable size fell from 4,136 to 4,106 lines, 6,127 lines below its 10,233-line baseline (59.9%); runtime
  remains 454 lines.
- Hardware replay remains deferred pending NPU recovery; this milestone did not access `/dev/dri`.

Verification:

- Three current isclose codegen variants: 3/3 accepted; old/forced-generic serialized images byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 97 passed in 4.48 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 149. Share the native-INT16 byte arena — complete

- Replaced three duplicated scratch-row allocators and constant-gather builders in INT32 division/remainder, bytewise
  logic, and barrel shifting with one `_int16_arena` physical helper. Each UOp handler still owns its real semantics;
  only physical row allocation and immutable constant materialization are shared.
- Compared serialized images before and after the rewrite for CDIV, CMOD, dynamic AND, XOR with a constant, constant
  SHL, and dynamic signed SHR. All six SHA-256 hashes and image byte lengths are identical, including the 140,443-byte
  restoring-divider images and their 3,479 EW stages.
- No CPU arithmetic, tensor-operation recovery, or generated hardware behavior changed.
- Renderer executable size fell from 4,106 to 4,092 lines, 6,141 lines below its 10,233-line baseline (60.0%); runtime
  remains 454 lines.
- Hardware replay remains deferred pending NPU recovery; this milestone did not access `/dev/dri`.

Verification:

- Old/current INT32 physical images: 6/6 byte-identical.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 97 passed in 4.43 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 150. Fail closed after an unrecoverable driver timeout — complete

- Diagnosed the OS crash from the previous-boot kernel journal. A long native chain timed out after completing 1,292
  tasks; the driver's own abort path then failed two NPU-IOMMU force resets and failed to reattach the IOMMU domain.
  A later GEM allocation entered `rknpu_iommu_dma_map_sg` with the detached domain and caused a kernel null-pointer
  dereference. This is a driver failure state, not a recoverable transient start timeout.
- Removed the runtime's four reset-and-retry attempts. A submit timeout is now attempted exactly once, marks the device
  poisoned, and raises an explicit platform-reset/power-cycle error. Later submissions, synchronizations, allocations,
  and reset ioctls fail before entering the damaged kernel path; cleanup unmaps userspace memory without issuing GEM
  destroy ioctls to the poisoned device.
- Found a separate hazard in `~/rk3588/examples/elementwise.py`: its terminal PC-chain descriptor writes
  `PC_BASE_ADDRESS=0`. Its 131,072-lane health case is the first three-task chain and timed out after all smaller cases
  passed. Tinygrad's runtime already avoids this RK3588 speculative-fetch hazard by terminating at a mapped zero-filled
  guard page, so no equivalent terminal-address bug remains in the production PC-chain builder.
- No CPU arithmetic or fallback was added. The change only prevents further driver access after hardware timeout.
- Runtime executable size is 453 lines, one line below milestone 149; renderer remains 4,092 lines.

Verification:

- Timeout regression: one submit ioctl, zero retries/resets, poisoned second submission rejected before ioctl.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 97 passed in 5.21s.
- `.venv/bin/python -m ruff check tinygrad/runtime/ops_rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/runtime/ops_rockchip.py`: pass.
- `git diff --check`: pass.

### 151. Bound native-INT16 PC chains before driver submission — complete

- Audited the commit history against the authoritative milestone-43 checkpoint, where the post-catalog renderer passed
  all 445 collected cases in one process: 433 passed, 12 skipped, zero failures, and 154 subtests passed. Later cleanup
  milestones did not repeat that complete gate until milestone 141, which already reported 12 regressions; the old
  checkpoint is therefore the behavioral oracle for the repair rather than evidence that missing UOp semantics are
  required.
- Found an unsafe physical-executor hole exposed by later generic lowering. Ordinary stateful groups are limited to 48
  tasks per PC chain, but the pure native-INT16 scratch fast path bypassed that limit and submitted its complete cached
  stage list at once. Previous crash journals show these long jobs timing out after 1,241 and 1,292 completed tasks,
  after which the kernel driver's IOMMU reset/reattach failure made later allocation capable of crashing the OS.
- Split cached native-INT16 stages at the existing 48-task physical limit. Every stage already carries complete stateful
  INT16 configuration, so the split changes only ioctl boundaries and preserves NPU arithmetic, dependencies, memory,
  and serialized RKImage semantics. No host tensor arithmetic or fallback was added.
- Added a host-independent runtime regression proving a 129-stage INT16 chain is submitted as 48, 48, and 33 tasks.
  Hardware replay is intentionally deferred: the latest reboot followed an unsafe platform recovery attempt, and this
  milestone does not open `/dev/dri`, reset the NPU, or risk another kernel failure.
- Runtime remains 453 executable lines because the bounded submit loop replaces the previous unbounded submit line;
  renderer remains 4,092 executable lines.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 98 passed in 4.78 seconds.
- `git diff --check`: pass.

### 152. Return to the zero-failure renderer before resuming deletion — complete

- Audited the full branch history after repeated NPU timeouts and confirmed two exact zero-failure baselines for the
  authoritative `test/backend/test_rockchip.py` census: `2f09e4a035` and the post-catalog-deletion `9c081e342` both
  completed 433 passes, 12 explicit skips, 154 subtests, and zero failures. The current file collects 445 pytest cases;
  future reports keep that count separate from subtests instead of calling the census 455.
- Preserved the pre-rollback cleanup tip at branch `rockchip-2608-ew-uops-cleanup-backup`, then restored only the
  Rockchip renderer, runtime, and their implementation-specific unit census from `9c081e342`. The current upstream
  ported-test file and complete branch history remain intact, so deleted-line patches can be recovered individually.
- Reapplied the two independently proven crash guards to the old runtime: native INT16 scratch chains are split at 48
  tasks, and a driver timeout poisons the device without retry/reset/free ioctls that can reach the failed IOMMU domain.
- The exact scatter-add-with-infinity codegen image is again byte-identical to the all-pass oracle
  (`839a7fa2f49dfd40db32e0ef926c91c9e00368cb9bf75489d3dbfac0ef857a38`). This reverses the later compensated-FP32
  storage path which turned infinite scatter contributions into NaNs.
- Renderer/runtime executable size returns to 6,658/487 lines. This deliberately gives back later cleanup savings so
  deletion can restart from a semantic baseline and each milestone can be checked against the complete census.
- No CPU numeric semantics were introduced. The rollback restores the already audited native DPU implementation and
  retains only submission safety changes.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 85 passed in 7.72 seconds.
- Exact scatter-add-with-infinity codegen image: byte-identical to `9c081e342`.
- `DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py`: pass.
- `git diff --check`: pass.

### 153. Rebuild the compact typed-UOp executor and isolate the remaining submit-state failure — in progress

- Created WIP checkpoint `ea8a95106` from the restored oracle and follow-up safety checkpoint `9d3bd5101`. The renderer
  is again a typed physical UOp executor: semantic UOps lower through `RKValue` layouts, static gathers, DPU EW recipes,
  and generic structural reductions. Runtime host work remains limited to address/layout materialization and raw lane
  copies; no CPU/GPU tensor arithmetic or numeric fallback was added.
- Restored the compact 4,092-line cleanup renderer, then repaired the known late-cleanup semantics without restoring
  tensor-operation lowerers. Non-finite FP32-to-half ADD trees avoid TwoSum's `inf-inf` NaN, repeated composed product
  terms use one generic mapped TwoProduct/Kahan recipe, and bounded exact INT32 expressions lower compositionally only
  when every converted intermediate is statically proven inside binary16's exact-integer range. Conditional STORE is
  accepted only when its Boolean gate is proven tautological; all other unsupported control flow still fails closed.
- The slow cumulative-index path now uses a generic scalar-local `RANGE/MAX` mapper. The exact 1,022-lane image is
  101,138 bytes with 1,106 logical EW stages, 12 initial gathers, 1,022 synchronized gathers, and no host-address work;
  its previous literal image had about 46,000 EW stages and a 4.1 MiB serialization. In the partial census,
  `test_simple_cummax` completed both sizes in 57.34 seconds and `test_simple_cummin` in 117.28 seconds, versus the
  earlier roughly 204-second cummin path. The expanded 512-term MAX spelling remains a separate unoptimized form.
- Hardened physical submission without retrying a failed driver: changed command/descriptor bytes receive fresh GEMs,
  every independent chain clears DPU/RDMA ping-pong pointers, native and spatial chains are both bounded to 48 tasks,
  and a timeout poisons the process before any further driver ioctl. The spatial-cap regression proves 49 operations
  over two hardware tiles submit as `[48,1,48,1]`.
- The first post-cap 445-case replay reached the upstream TopK method before the first real failure. Pytest recorded
  190 passes, 6 explicit skips, and 126 passing subtests; TopK then timed out at task counter zero on a two-task index
  chain, and the following 11 failures were only the deliberately poisoned process. The run was stopped rather than
  accessing the driver again, so this is not an all-pass result and the unreplayed tail is not counted as passing.
- Offline reconstruction found a physical regression rather than a renderer/UOp error. The repeated TopK images match
  the 4,092-line cleanup image exactly, while the new rearm prefix unconditionally wrote
  `DPU_DST_SURF_STRIDE=0`. A terminal native-INT32 chain follows an INT16-to-INT32 stage that establishes the required
  `0x10` four-byte output-surface stride; clearing it immediately before the native-INT32 task explains the zero-task
  start failure. The current WIP derives the rearm stride from DATA_FORMAT output precision: INT32/FP32 outputs use
  `0x10`, while FP16/INT16 outputs, including INT32-to-FP16, use zero.
- Hardware confirmation is pending an external power cycle. The required vendor health probe did not reach arithmetic:
  its 4 MiB allocation exhausted CMA, the non-contiguous mmap fallback returned `ENXIO`, and process teardown logged a
  DRM GEM refcount underflow. No TopK retry or further `/dev/dri` access was attempted after that kernel warning.
- Current executable size is renderer/runtime **4,340/507** lines, total **29,914**. This is 5,893 renderer lines below
  the 10,233-line baseline but still 340 lines above the 4,000-line target; neither the size target nor the 445-case gate
  is complete.

Verification at this WIP boundary:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 108 passed in 4.88 seconds.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.
- Full Rockchip census: not passed; the partial result above is retained as the exact current boundary.

### 154. Compose wide INT32 division as ordinary UOps — WIP checkpoint

- Removed the 49-line whole-root division/floor-dialect recognizer and its 155-line physical lowering catalog. Wide
  `CDIV` and `CMOD` now lower at their true UOp boundary inside `RKContext`, so nested operands, sibling quotient and
  remainder consumers, comparisons, `WHERE`, and Tinygrad's ordinary floor corrections compose without recovering a
  tensor or root-graph operation.
- The handler keeps one byte-restoring divider state per ordered operand pair. It splits canonical INT32 values into
  widened unsigned-byte lanes, executes sign/magnitude conversion and restoring division with native INT16 DPU EW,
  caches quotient and remainder byte components, and packs only the requested results back into canonical INT32.
  Sibling `CDIV`/`CMOD` UOps therefore share the 32-step restoring core rather than duplicating it.
- The proven-bounded `INT_FP16` path remains separate and unchanged. Division by zero keeps the existing Tinygrad
  contract (`quotient=0`, `remainder=lhs`), while `INT_MIN/-1` and other overflow cases wrap at the canonical 32-bit
  pack boundary. There is no host tensor arithmetic, graph-dialect recovery, or CPU/GPU fallback in production code.
- Added a test-only physical RKImage interpreter for raw gathers and the exact signed INT16/INT32 EW subset used by the
  divider. It executes direct and composed `CDIV`/`CMOD`, the legal INT16-to-INT32 boundary, and ordinary floor/floormod
  correction graphs over 100 deterministic random and edge-case lanes. This is verification code only; production
  arithmetic remains entirely on the DPU.
- Renderer executable size fell from **4,340 to 4,212 lines** (net -128); runtime remains **507** lines and total tree
  size is **29,786**. The renderer remains 212 lines above the first 4,000-line target.
- Hardware validation remains pending the already-required manual power cycle. This checkpoint did not open `/dev/dri`
  and does not claim the 445-case census passes.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 113 passed.
- Test-only physical semantics: direct and composed CDIV/CMOD plus floor/floormod passed signed, zero-divisor,
  `INT_MIN/-1`, and randomized cases.
- Adversarial verification: `VERIFIED` after the physical floor/floormod coverage gap was closed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `test/backend/test_rockchip.py --collect-only`: 445 tests collected.
- `git diff --check`: pass.

### 155. Unify typed raw components and compose INT32 bitwise UOps — WIP checkpoint

- Removed the root-only wide-INT32 `AND`/`OR`/`XOR` lowerer and its early dispatch. These operations now lower at
  their own UOp boundary through `RKContext`, so direct and nested operands share the same physical byte-plane ABI as
  comparisons, `WHERE`, `NOT`, division, and later consumers. Production code emits only static gathers and DPU EW;
  it adds no host tensor arithmetic, CPU/GPU fallback, or recovered tensor-operation dialect.
- Consolidated the separate FP16, INT16, and INT32 raw split/pack paths into typed `_raw_parts` and `_pack_raw`
  helpers. The wide-INT32 pre-copy and raw-`WHERE` scheduling rules are preserved, and FP16 and INT32 less-than now
  reuse one ordered-byte comparison core. The superseded raw caches/helpers and legacy bitwise dispatch have no
  remaining references.
- Added physical semantic coverage for direct and nested `AND`, `OR`, and `XOR` across deterministic random lanes and
  edge values including `INT_MIN`, `INT_MAX`, `-1`, zero, one, and alternating masks. The emitted images remain
  `NATIVE`, contain no host gather/scatter, and round-trip through the RKImage encoder. The 64,000-byte-lane hardware
  limit is admitted while 64,004 byte lanes fail closed.
- Representative non-migrated images across FP16/INT16 comparison and packing, INT32 comparison and `NOT`, raw
  `WHERE`, `CDIV`, and `CMOD` remained byte-identical. The migrated root bitwise images have the expected small
  synchronization-layout change and passed the physical interpreter.
- Renderer executable size fell from **4,212 to 4,131 lines** (net -81); runtime remains **507** lines and total tree
  size is **29,705**. The renderer is still 131 lines above the first 4,000-line target.
- Hardware validation remains pending the required manual power cycle. This checkpoint did not open `/dev/dri` and
  does not claim that the 445-case census passes.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 113 passed.
- Independent constant-operand and maximum-lane bitwise matrices: passed.
- Adversarial verification: `VERIFIED WITH CAVEATS`; the caveats are deferred hardware validation and the finite
  scope of serialized-image sampling, with no contradictory image found.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `git diff --check`: pass.

### 156. Compose signed and unsigned INT32 shift UOps — WIP checkpoint

- Removed the root-only 32-bit shift planner, its early dispatch, and the private INT16 arena that only that planner
  used. `SHL` and `SHR` now lower at their own UOp boundary inside `RKContext`, so shifts can consume and feed ordinary
  INT32 arithmetic and bitwise UOps without recovering a root graph shape.
- The physical recipe keeps the five-stage byte-plane barrel shifter: canonical four-byte `RKValue` operands are split
  into exact native-INT16 bit planes, batched across 32 rows, shifted according to the low five amount bits, and packed
  back to the canonical INT32 ABI. Signed `SHR` replicates the sign plane; unsigned `SHR` inserts zero. A minimal raw
  UINT32 load/constant/static and UINT32-to-INT32 cast bridge preserves upstream logical-shift behavior while leaving
  unrelated UINT32 ALU unsupported and fail-closed.
- Replaced the old terminal-gather shape assertion with physical semantics across signed and unsigned `SHL`/`SHR`,
  constant and tensor amounts `0/7/8/15/16/31/32`, random and edge values, nested shift consumers, and shifts whose
  value is itself another shift. All images are `NATIVE`, have no host gather/scatter, and round-trip through RKImage
  serialization. Shift amount 32 retains the existing raw modulo-32 behavior.
- Independent adversarial verification exercised 92 physical cases plus aligned stripe boundaries: 1,984 values
  lower successfully, while 1,985 reject at the 64,000-native-lane limit. Signed arithmetic right shift, unsigned
  logical right shift, left-shift wrapping, and nested composition matched exact 32-bit reference results.
- Representative direct codegen grows only by the generic raw-value copy: signed constant `SHL` changes from 94 to 95
  EW stages, and tensor `SHR` from 94 to 96 stages. No CPU/GPU numeric path or runtime host arithmetic was added.
- Renderer executable size fell from **4,131 to 4,095 lines** (net -36); runtime remains **507** lines and total tree
  size is **29,669**. The renderer is 95 lines above the first 4,000-line target.
- Hardware validation remains pending the required manual power cycle. This checkpoint did not open `/dev/dri` and
  does not claim that the 445-case census passes.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 113 passed.
- Adversarial verification: `VERIFIED` with no load-bearing defect.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `test/backend/test_rockchip.py --collect-only`: 445 tests collected.
- `git diff --check`: pass.

### 157. Use guarded canonical UOp bounds and one image-alias primitive — WIP checkpoint

- Replaced the backend's independent recursive integer range evaluator with Tinygrad's canonical UOp `vmin`/`vmax`
  contract behind a strict topology guard. Only the previously supported CONST, static RANGE/SPECIAL, integer/bool
  CAST, WHERE arms, bitwise-NOT XOR, constant-divisor CDIV/CMOD, and ADD/SUB/MUL/MAX shapes are admitted. Dynamic LOAD
  expressions and unsupported operations still return unknown instead of inheriting a dtype-wide bound.
- Every admitted node is checked against its scalar dtype before its bounds can select FP16/INT16 storage. Oversized
  RANGE/SPECIAL roots and their ancestors therefore fail closed, while legal INT32 endpoints and the exact
  `±2,048`/`±32,768` layout boundaries remain unchanged. The first adversarial review found the missing root
  RANGE/SPECIAL check; the corrected implementation and ancestor-poisoning regression then received `VERIFIED`.
- Added `_alias_image_args` and replaced six local ARG-to-scratch/ARG remapping closures used by product residuals,
  mapped ADD loops, repeated product reductions, scalar-local MAX, and multi-local reductions. It preserves buffer
  kind, target base addend, and the original argument byte addend. Representative mapped/reduction images remained
  byte-identical across the consolidation.
- Renderer executable size fell from **4,095 to 4,064 lines** (net -31): guarded UOp bounds save 19 lines and shared
  image aliasing saves 12. Runtime remains **507** lines and total tree size is **29,638**. The renderer is 64 lines
  above the first 4,000-line target.
- Hardware validation remains pending the required manual power cycle. This checkpoint did not open `/dev/dri` and
  does not claim that the 445-case census passes.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 116 passed.
- Randomized supported-expression differential: no new unsafe topology acceptance or range.
- Representative remapped images: byte-identical before/after.
- Adversarial verification after the overflow fix: `VERIFIED`.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `test/backend/test_rockchip.py --collect-only`: 445 tests collected.
- `git diff --check`: pass.

### 158. Cross the first 4,000-line target with canonical static ALU and typed comparison composition — WIP checkpoint

- Replaced the backend's hand-written compile-time ALU dispatch with Tinygrad's canonical `exec_alu`, `python_alu`,
  and `float_to_fp16` primitives. Rockchip retains only the cases whose scalar/vector contracts differ: lazy scalar
  versus eager vector `WHERE`, NumPy-safe vector division/modulo and `MAX`, zero-divisor behavior, and explicit dtype
  recasting. This evaluator is still used only for compile-time static indexes, masks, constants, and layout plans; it
  does not read runtime tensor values or execute runtime arithmetic on the host.
- Consolidated native-INT16 comparison emission behind `RKContext._i16` and `_i16_equal`. FP16 zero canonicalization,
  NaN classification, byte ordering, equality, and wide-INT32 comparison now compose through the same typed physical
  primitives instead of parallel global stage builders. The emitted scratch allocation, EW ordering, flags, and
  serialized images remain unchanged for all nine focused FP16/INT32 comparison and mixed-`WHERE` oracles.
- Deleted `_fold_general_where` and its sole fallback matcher. The renderer already attempts the generic typed UOp
  executor before FP16 recipe rewriting, and `RKContext` owns true ternary `WHERE` semantics, including non-finite
  arms. A regression proves the recipe pass leaves an ordinary finite `WHERE` intact and the recipes-ready generic
  executor lowers it natively.
- Renderer executable size fell from **4,064 to 3,989 lines** (net -75): canonical static evaluation saves 17 lines,
  and typed comparison/`WHERE` consolidation saves 58. Runtime remains **507** lines and total tree size is **29,563**.
  This crosses the first 4,000-line target without deleting comments/docstrings or restoring a tensor-operation
  catalog.
- Hardware validation remains pending the required manual power cycle. This checkpoint did not open `/dev/dri` and
  does not claim the 445-case hardware census passes; it is the requested WIP commit boundary before that replay.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 118 passed.
- Static-evaluator differential: 18,707 exact cases; 151 old/current images byte-identical and four identical
  unsupported outcomes across 2,762,522 encoded bytes.
- Nine FP16/INT32 comparison and mixed-`WHERE` images: byte-identical, `NATIVE`, and zero host gathers/scatters.
- Independent adversarial verification: `VERIFIED`; no weakened test, host numeric path, orphaned symbol, or debris.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.

### 159. Execute static local programs through one ordered symbolic path — WIP checkpoint

- Removed the separate NumPy-based `_static_local_load_offsets` address interpreter and its private `RKContext`
  offset side channel. Static local buffers, dependent local buffers, indexes, and load gates now expand into ordinary
  UOps under their inherited RANGE environment and then use the same typed `_gather_plan` path as every other load.
- Replaced three local-load DAG walkers with one semantic traversal that ignores `AFTER` ordering-only sources while
  preserving stable first-use order. Identity-indexed local bridges, scalar mapped MAX, and multi-local ADD retain
  their existing physical image contracts; representative nested-address and mapped-local images remained
  byte-identical.
- The first consolidation incorrectly rebuilt local updates with balanced `_structural_reduce`. Adversarial review
  exposed the FP16 sequence `[2048, 1, -2048]`: a STORE loop rounds sequentially to zero, while the balanced tree
  produced one and changed a source address or load gate. The accepted implementation uses an ordered left fold and
  restores the aggregate `1 << 20` expansion budget. Both counterexample images are again byte-identical to the
  pre-change renderer, 107 bytes, `NATIVE`, one EW stage, offset zero, and no host execution.
- Renderer executable size fell from **3,989 to 3,894 lines** (net -95); runtime remains **507** lines and total tree
  size is **29,468**. Hardware validation and the full 445-case execution remain pending the required manual power
  cycle, so this checkpoint makes no all-pass claim.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 121 passed.
- Sequential FP16 address/gate images: byte-identical to `ee3777470`, with offsets `(0,)`.
- Independent adversarial verification: `VERIFIED WITH CAVEATS`. The caveat is a pre-existing, unchanged
  multi-scalar local ADD path that uses compensated mapped reduction instead of literal sequential FP16 updates; it
  remains a separate fail-closed/semantic follow-up rather than an overclaim in this checkpoint.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.

### 160. Map repeated integer MAX and cross the 3,800-line target — WIP checkpoint

- Added a generic lowering for a top-level plain integer `MAX` tree whose 128–512 terms can be proven to be one
  affine UOp template. The renderer anti-unifies only differing typed integer constants, reconstructs and
  symbolically normalizes every term, and accepts the mapping only when all original UOp keys match. Existing
  nonnegative and exact-FP16-integer bounds still gate the physical mapped-MAX executor; malformed, negative,
  over-2,048, floating-point, 127-term, and 513-term forms fail closed.
- This removes the current slowest renderer/runtime shape without naming cumulative extrema. The 512-element cummin
  index image fell from 23,064 to 598 logical EW stages and now predicts 1,033 physical EW tasks; cummax fell from
  22,552 to 597 stages and 1,028 tasks. Their encoded images are 50,640 and 50,600 bytes. The existing 1,022-term
  scalar-local image remains byte-identical at 101,138 bytes, 1,106 EW stages, and 1,022 synchronized gathers.
- Consolidated generic and non-finite `WHERE` execution into one lazy raw-bit selector and removed the unreachable
  EXP2-by-infinity recognizer. Root-INT, static-root, and ABS/ordered/threshold fast paths remain. Independent review
  exhaustively selected all 65,536 FP16 bit patterns under both BOOL layouts and found the relevant existing
  static, threshold, non-finite, and EXP2 images byte-identical to the previous checkpoint.
- Deleted the inactive multi-scalar local ADD specializer. None of five real `std_mean` spellings entered it, while a
  synthetic pair of FP16 STORE loops exposed that it reassociated `[2048, 1, -2048]` into two instead of preserving
  the sequential result zero. The ordinary symbolic path now emits the correct native 72-byte, one-EW zero image;
  the real `std_mean` image acceptance/rejection and hashes remain unchanged. Four trivial static-evaluator/gather
  forwarding wrappers were also inlined without changing their physical consumers.
- Renderer executable size fell from **3,894 to 3,799 lines** (net -95); runtime remains **507** lines and total tree
  size is **29,373**. This crosses the requested 3,800-line checkpoint while also addressing the measured slowest
  image. Hardware validation and the full 445-case execution remain pending the required manual power cycle, so this
  WIP checkpoint does not claim a passing backend.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 126 passed.
- Repeated-MAX image contracts: exact cummin/cummax hashes, counts, physical-task estimates, native execution, and no
  host gather/scatter; the prior 1,022-term image is byte-identical.
- Raw `WHERE`: exhaustive 65,536-pattern selection under BOOL_MASK and BOOL_INT16; 24 focused old/current images are
  byte-identical.
- Independent adversarial verification: `VERIFIED WITH CAVEATS`; only hardware execution and exhaustive enumeration
  of every conceivable hand-authored EXP2 graph remain unverified.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.

### 161. Restore linear synchronized-gather scheduling — WIP checkpoint

- Profiling after the 3,800-line checkpoint again identifies dynamic `nonzero` as the next slow backend family. Its
  runtime revisited every mid-program gather for every synchronization point, even though each gather already carries
  its exact physical `after` boundary.
- Restored the previously proven stable grouping pass from `7490132d0`: build each `after` batch once, then execute the
  batches in sorted boundary order. Gather order within a boundary, EW slices, device synchronization, buffer clearing,
  and raw memory movement are unchanged. This is scheduler metadata work only; it reads no tensor value and adds no
  CPU/GPU numeric semantics or CMAC path.
- The original hardware A/B for this exact transformation measured 9.60 seconds in the quadratic grouping code and
  reduced cold `TestRockchipNonzeroOps::test_nonzero` from 35.00 to 28.34 seconds. Those numbers are historical evidence,
  not a fresh device result: this checkpoint deliberately did not open `/dev/dri` while the manual power cycle remains
  pending.
- Renderer executable size remains **3,799 lines**. Runtime grows from **507 to 508 lines** and total tree size from
  **29,373 to 29,374**. The next renderer milestone still targets 3,600 lines and the repeated predicate/selection
  programs that dominate the remaining `nonzero` work.
- Hardware validation and the all-445 replay remain pending the required manual power cycle, so this checkpoint does
  not claim a passing backend.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 126 passed.
- Independent adversarial verification: `VERIFIED`; 2,396,745 old/new scheduler traces matched exactly.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.

### 162. Retire superseded fallback recipes and cross the 3,600-line target — WIP checkpoint

- Removed the FP16 fallback recipes that the generic typed-UOp executor now bypasses for real cosine, tangent,
  round/floor/ceil/trunc, sign, inverse-hyperbolic, gradient, and remaining math/storage graphs. The underlying
  compositional helpers remain wherever generic execution calls them directly. The fallback pass now contains only
  the observed bool-to-half conversion; generic storage lowering retains its required ABS and infinite-multiply
  callbacks.
- Added a focused first-pass ownership regression for cosine and tangent, including stable native image hashes. An
  independent committed/current compiler replay compared 40 representative images and all 402,474 encoded bytes were
  identical. Every current image was produced by the first generic lowering attempt rather than the deleted fallback
  catalog.
- Consolidated EW-stage tuple construction and the Boolean tautology/structured-STORE proof without changing their
  contracts. Independent randomized comparison matched 20,000 EW stages, 50,000 Boolean DAGs, and 5,000 structured
  scopes plus 11 malformed-scope cases exactly.
- Renderer executable size fell from **3,799 to 3,598 lines** (net -201); runtime remains **508** lines and total tree
  size is **29,173**. This crosses the requested 3,600-line checkpoint without deleting comments/docstrings, adding
  host tensor arithmetic, restoring a tensor-operation catalog, or introducing CPU/GPU/CMAC execution.
- Independent adversarial verification returned `VERIFIED WITH CAVEATS`. The only caveat is intentional: the NPU was
  not accessed while the manual power cycle remains pending, so the complete 445-case hardware census has not run and
  this checkpoint makes no all-pass backend claim.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 127 passed.
- Committed/current real-image replay: 40/40 byte-identical images across 402,474 encoded bytes.
- Structural differentials: 20,000 EW stages, 50,000 Boolean DAGs, 5,000 valid scopes, and 11 malformed scopes matched.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.

### 163. Superseded static-local renderer experiment — corrected checkpoint

- The 73-UOp static-local fixture and its 66-byte identity image were valid renderer-only measurements, but follow-up
  whole-pipeline profiling showed that the real scalar-true graph never reached this stage. `devectorize2` received
  2,064 nodes and expanded beyond 166,000 visited nodes and 174,000 replacements in a bounded 11.8-second trace; an
  unbounded probe exceeded 45 seconds and 2.4 GiB without producing renderer UOps.
- The bounded vector evaluator therefore did not solve the actual workload. Its +99 renderer lines and focused tests
  are removed. The separate sequential `_unroll_static_local` preflight remains because it prevents environment-list
  construction beyond the existing step budget.
- Five initial diagnostic `size=None` probes may have indirectly realized small Rockchip count kernels before the
  count was stubbed. They are not hardware evidence. Later profiling used an invalid DRM path and an item stub.

### 164. Fold scalar-true selection and consolidate dynamic typed loads — WIP checkpoint

- `masked_select(size=None)` with a rank-0 direct constant-true mask now returns `self.flatten()` at the semantic API
  boundary. This avoids constructing the cumulative-sum/scatter graph, takes roughly 0.1 ms in the compile-only probe,
  and schedules no compute SINK. Tensor and bare-UOp identity tests cover multidimensional, empty, and scalar inputs;
  fixed-size, scalar-false, rank-one, and derived-true masks remain unchanged. The Rockchip submit expectation changes
  from one to zero while value equality remains required.
- Consolidated the 76-line raw candidate emitter and 63-line dynamic typed-load parser into one candidate-major native
  lowerer. It preserves FP16 raw payloads, INT16/INT32 bytes, negative normalization, external Boolean gates, multiple
  axes, repeated channels, odd reductions, and blocked 1,001-candidate selection. A test-only physical interpreter
  exercises all of these contracts, including exhaustive valid/no-match selection for the blocked fixture.
- The new physical shape retains logical EW counts while reducing gathers: representative INT32 selection falls from
  77 to 13 gathers, the two-axis fixture from 1,477 to 36, and the blocked fixture from 6,016 to 22. The independent
  bounded-coordinate path remains byte-identical at 5,337 bytes, SHA-256
  `065ebb556535c2724c9b7a641fb13a7bd464ce0a119bb4c61a1cdcf0f370f11c`.
- Adversarial review caught signed-i32 byte-address overflow and u16 image-table overflow in the first consolidation.
  The final admission rejects all oversized ARG slots and conservative scratch/gather plans before static evaluation,
  checks actual index/source byte offsets before physical allocation, and audits the final specialized image. Element
  offset `2^29-1` round-trips for INT32 data and indices; `2^29` rejects before gather construction. Eight older direct
  specialized returns do not all share this centralized audit and remain a separate follow-up caveat.
- Renderer executable size is **3,595 lines**, down 102 from the superseded WIP and three below milestone 162; runtime
  remains **508**, `mixin/op.py` grows by one line, and total production size is **29,171** (net -101).

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 139 passed.
- `.venv/bin/python -m pytest test/null/test_tensor_uop_mixin.py -x -q -n12`: 216 passed.
- Scalar-true CPU/Torch and JIT focused tests: passed; compile-only Tensor/UOp identities schedule zero compute SINKs.
- Dynamic selector physical simulation: raw FP16/INT16/INT32, normalization, gates, multi-axis/repeat/odd/blocking cases
  passed; 1,001-candidate image remains 724,098 bytes, 22 gathers, and 2,042 EW stages.
- Independent adversarial verdicts: scalar identity `VERIFIED`; selector boundary hardening `VERIFIED WITH CAVEAT` for
  the eight pre-existing specialized early-return paths outside this change.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `ALLOW_DEVICE_USAGE=0 DEV=NPY .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.
- Hardware execution remains pending the required manual power cycle; no all-445 pass is claimed.

### 165. Keep oversized scaled sums on the mapped reduce loop — WIP checkpoint

- Compile-only profiling after the 3,595-line checkpoint, using NPY tensors and `RockchipRenderer` without opening
  `/dev/dri`, ranked `test_nll_loss_3d` as the slowest remaining family: 29.4 seconds, 20,642 linearized UOps,
  148,597 EW stages, 2,594 gathers, 8,648 mid-gathers, and 864 host gathers. `log_softmax` kernels stayed compact
  (67/117 UOps); the scalar mean of the class gather exploded.
- The Rockchip hand-coded unroll treated the documented 512-term scaled-sum budget as a per-axis limit. Axes 32, 3, 3,
  and 3 each fully unrolled. The heuristic now skips unroll when a scaled ADD's reduce-axis product exceeds 512, so
  the existing mapped-add loop specializer keeps a 32×27 REDUCE. Non-scaled reductions, including 512-element cummin
  MAX, retain the old per-axis `else 32` path.
- The 3D NLL mean kernel is now 63 UOps, 986 EW stages, six gathers, 882 mid-gathers, and one `HOST_ADDRESS` gather for
  the INT32 class index. Compile-only wall time is 0.59 seconds. 512-element mean (2.16 s, 9,711 EW, NATIVE), 5×5
  average pool, and 32-element 2D NLL images are unchanged. Weighted 3D NLL is 16×27=432 ≤ 512 and still unrolls
  (~4.5 s); that remains a follow-up under the same budget.
- Renderer executable size remains **3,595**; `tinygrad/codegen/opt/heuristic.py` grows from **143 to 144** and total
  production size is **29,172** (net +1). This is a performance WIP, not the 3,400-line cleanup milestone.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 140 passed.
- `test_nll_loss_3d_mean_keeps_mapped_reduce_loop`: mapped loop, <200 UOps, <2,000 EW stages, ≤1 host gather.
- Exact 512-element cummin/cummax images: byte-identical to the previous checkpoint.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `ALLOW_DEVICE_USAGE=0 DEV=NPY .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.
- Hardware execution remains pending the required manual power cycle; no all-445 pass is claimed.

### 166. Map large ADD reduces and weighted NLL FDIV — WIP checkpoint

- Compile-only profiling after 165, without opening `/dev/dri`, ranked `std_mean` k1 reject (~10 s), 512-element
  cummin rewrite (~8.8 s), and `test_nll_loss_3d_weight` as the slowest successful family: 4.8 s, 83 UOps after the
  product>512 skip, but generic lowering still expanded 16×27=432 gathers into 82,952 EW stages and 432 host gathers.
  512-element mean/sum were 2.1–2.4 s on fully unrolled ADD trees.
- The Rockchip heuristic now skips unroll for every ADD whose reduce-axis product exceeds 64. MAX keeps the per-axis
  `else 32` path, so 512-element cummin/cummax images stay byte-identical. 5×5 average pool (25) and 2D NLL (32) still
  unroll.
- `_lower_mapped_add_loop_reduction` now maps each local ADD on its own index ranges, including FDIV of two locals.
  Range cleaning drops END/AFTER parents so a trailing 432-weight sum is not flattened as 16×27×432. CAST-wrapped
  `sum*1/n` peels to `post_scale`. `_append_inplace_image` preserves the first image's host gathers.
- Weighted 3D NLL compile-only: 0.78 s, 987 EW stages, one host gather. Mean-512: 0.07 s, 512 EW. Sum-512: 0.04 s.
  Unweighted 3D NLL remains mapped (554 EW, one host gather). Horner/`_half_storage` helpers recover some renderer
  lines. LUT/PPU were not adopted; they add encoder surface.
- Renderer executable size **3,595 → 3,586**; `heuristic.py` **144 → 142**; total production size **29,172 → 29,161**.
  This is a performance WIP, not the 3,400-line cleanup milestone.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 142 passed.
- `test_nll_loss_3d_weight_keeps_mapped_reduce_loop` and `test_mean_512_keeps_mapped_reduce_loop`.
- Exact 512-element cummin/cummax and cos/tan images: byte-identical to the previous checkpoint.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `ALLOW_DEVICE_USAGE=0 DEV=NPY .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.
- Hardware execution remains pending the required manual power cycle; no all-445 pass is claimed.

### 167. Keep large MAX on the mapped reduce loop — WIP checkpoint

- Compile-only profiling after 166, without opening `/dev/dri`, ranked 512-element `cummin` indices as the slowest
  remaining family: 8.84 s, 5,623+6,157 linearized UOps. The mapped-MAX specializer already compact the image (598 EW),
  but rewrite still expanded the fully unrolled 512-term MAX tree. `std_mean` now succeeds (4.26 s) via mapped-add of
  13,125 lanes; it is no longer a reject.
- The Rockchip heuristic now skips unroll for ADD and MAX whose reduce-axis product exceeds 64. Small MAX still unrolls:
  32-element max, 2×2 pool. The existing local-loop branch of `_mapped_max_descriptor` owns 512-element prefix extrema.
- Compile-only `cummin` indices: 8.84 s → 1.25 s, 48 UOps, 596 EW stages. `cummax` indices: 7.87 s → 0.63 s. The 1,022-term
  synthetic scalar-local image and cos/tan hashes stay byte-identical. Cumulative-index exact images are updated to the
  loop-mapped native forms and the helper no longer constructs `device="ROCKCHIP"` tensors.
- Renderer compaction is incidental (`finish` copy-out, `_fp32_expr_to_half`, `_int32_shift` gathers, `_alu` native cfg).
  LUT/PPU were not adopted; they add encoder surface versus existing polynomials and generic pooling UOps.
- Renderer executable size **3,586 → 3,564**; `heuristic.py` remains **142**; total production size **29,161 → 29,139**.
  This is a performance WIP, not the 3,400-line cleanup milestone.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 142 passed.
- Exact 512-element cummin/cummax images: 50,554/50,514 bytes, 596/595 EW, mapped-max loop, <80 UOps.
- 1,022-term scalar-local and cos/tan images: byte-identical to the previous checkpoint.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
- `ALLOW_DEVICE_USAGE=0 DEV=NPY .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q`: 445 tests collected.
- `git diff --check`: pass.
- Hardware execution remains pending the required manual power cycle; no all-445 pass is claimed.
