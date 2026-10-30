# Gemma CPU attention dispatch

Gemma3n and Gemma4 have their own shared-KV attention method. Previously,
Rockchip placement installed the ordinary CPU hooks, but that method never
called them. Recorded native traces contain zero CPU attention calls despite
installation counts of 30 and 28. Gemma4's seven full-attention blocks also
have 512-channel heads, beyond the old CPU kernel's 256-channel limit.

The Gemma method now consumes its hooks after Q/K normalization and RoPE and
the normalized-V half-precision KV store. Shared blocks receive the same
earlier cache and its ordering dependency. The CPU kernel supports a fixed
positive score scale; Gemma binds 1 rather than the default 1/sqrt(head_dim).
The causal window includes exactly `position-window < key <= position`.
Windowed prefill uses the per-row kernel; full causal prefill retains blocked
queries. Grouped decode shares KV loads across query heads. A window covering
the allocated context selects the full causal kernel.

Gemma's incompatible prepared-QKV path is excluded. The alternative packed
CPU kernel remains limited to its original scale and full causal mask.

An isolated CPU comparison used eight query heads, one KV head, a 512-token
cache and end position 511. Each variant had three warmups followed by 20
alternating ABBA/BAAB blocks (40 calls per variant).

| Rows | Head channels | Window | Generic ms | Custom ms | Paired speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 256 | Full | 6.136 | 0.526 | 11.576× |
| 1 | 512 | Full | 10.667 | 0.803 | 13.158× |
| 128 | 256 | Full | 653.924 | 10.855 | 60.289× |
| 128 | 512 | Full | 1,302.262 | 20.714 | 62.613× |
| 17 | 512 | 64 | 173.589 | 1.330 | 132.135× |

Every paired block favored the custom kernel. Maximum absolute output error
against the generic CPU path was below 3e-8. These are cached attention
fixtures, not whole-model TTFT or decode measurements.

Existing and added CPU suites passed 68 tests with two skips. They include
independent float64 attention, poisoned masked cache entries, positions near
8,192, dynamic rows, full-model shared-KV/normalization/reference checks, reset
and cached continuation. A separate 512-channel windowed prefill check matched
pooled and serial results bit for bit. Mypy and Ruff passed.

[Raw primitive samples](gemma_cpu_attention_results.json) retain the fixture,
timing and source identities. Full-model hardware runs and RKLLM quality
qualification are separate.
