# CPU final-logit soft cap

`--cpu-softcap` replaces the final CPU logit transform with a vector kernel on
bounded FP32 tensors. The model still computes `cap * tanh(logits / cap)`; the
kernel preserves the existing FP32 sigmoid-based tanh evaluation and its
first-index greedy ties. The option is disabled by default.

The fast path accepts static CPU tensors containing 1,024 to 4,194,304 values
and a finite cap between 0.125 and 128. Other supported tensors and positive
caps retain the generic expression. A zero model cap leaves logits unchanged.
Large vectors use four pthread workers when process affinity permits four
CPUs. Smaller vectors use one worker, and vector tails retain every element.

A 262,144-logit CPU fixture, including concatenation and dispatch, measured
4.727 ms for the generic expression and 1.144 ms for the vector path. All 20
alternating paired blocks favored the candidate, with a median paired speedup
of 4.061. Outputs matched the generic path bit for bit. This is an operator
measurement, not a complete-model speedup.

Runtime checks cover random inputs, nonfinite values, signed zero, tails,
noncontiguous inputs, fallback shapes and dtypes, parallel execution, and
saturated greedy ties. Small Gemma3n and Gemma4 reference models exercise
prefill, decode, reset and generation through the optional hook. The generic
fallback is checked bit for bit against its original expression; independent
NumPy comparisons allow the FP32 cancellation error of that expression.

```sh
DEV=CPU python -m pytest test/runtime/test_rockchip_softcap.py \
  test/models/test_gemma_softcap.py -q -n12 --dist=loadfile
```

Full-model timing, output-quality qualification and long-context qualification
are separate from these operator and small-model checks.
