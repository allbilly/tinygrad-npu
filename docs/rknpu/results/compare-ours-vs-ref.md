# Ours vs the reference — superseded

This file used to carry the full comparison. Its conclusion — "1.2 s is not reachable by any
combination of the rows this tree can still change" — was **wrong**: the four-way quantise lever
landed after it was written and prefill is now **1,198.87 ms**.

Its numbers are from the 1,402.60 ms tree, before the replay and the parallel quantise.

Current, per-file:

| question | file |
| --- | --- |
| what is the prefill result, with protocol and temperatures | `prefill-parallel-quantise.md` |
| which levers landed, which were refuted, and why | `../refuted-levers.md` |
| what the reference is and how to run it | `../../reference-qwen35-4b.md` |
| the attribution of our 1,198.87 ms | `int8-name-reconciliation.md` §1-§3 |

The old text is in git history if a derivation is ever needed. Its two durable facts were carried
forward: the `ffn_down` `m_rows` out-of-bounds bug (rows 64-127 were wrong in every number before
the 1,410.58 ms tree) and the list of refuted levers, now one line each in `../refuted-levers.md`.
