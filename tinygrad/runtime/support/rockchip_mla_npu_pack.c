// Host-side pack/unpack kernels for vendor-style FP16 NPU MLA prefill attention.
//
// Layouts (all little-endian fp16 unless noted):
//  qa  row-major QK LHS        (HEADS, capacity, 96) from q_nope(64)+q_rope(32) f32
//  ka  QK RHS N16K32           (HEADS, keys/16, 96/32, 16, 32) from k_nope(head-major f32)+k_rope f32
//  va  PV RHS N16K32           (HEADS, 64/16, keys/32, 16, 32) from v (head-major f32, token-major keys)
//  prob PV LHS row-major       (HEADS, capacity, keys) fp16: exact-order causal softmax of scores f32
//  exp expansion LHS row-major (HEADS, capacity, 256) fp16: replicate c_kv token-major f32
//  pv  PV result unpack        (capacity, HEADS*64) f32 token-major
//  wkb expansion K weight      (HEADS, 64/16, 256/32, 16, 32) from attn_k_b (HEADS,256,64) fp16
//  wvb expansion V weight      (HEADS, 64/16, 256/32, 16, 32) from attn_v_b (HEADS,64,256) fp16
//
// The exact-order softmax reproduces the reference CPU attention arithmetic (see
// rockchip_mla_attention.c): causal end=r+1, sequential max, (s-max)*log2e, the same exp2
// polynomial, sequential sum in the same order, divide, pad with +0.  The generated program
// source prepends `#pragma STDC FP_CONTRACT OFF` so clang cannot contract the polynomial or
// the divide into FMAs (8/655360 probabilities differ by 1 fp16 ulp otherwise).
#include <stdint.h>
#include <stddef.h>
