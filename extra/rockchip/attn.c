/* fp32 causal GQA attention for the CPU host: 16 heads / 4 kv heads / 256 dim.
 *
 * The generated kernel this replaces costs 252 ms per block at B=1,T=128 -- 1.06 GFLOP/s against
 * the ~11 GFLOP/s these cores reach. This is a straightforward float4 implementation at 15.28
 * GFLOP/s, 17.54 ms per block, within 5.0e-06 of an fp64 reference.
 *
 * Interface note: the KV side is the *whole* cache buffer (2, B, KVH, CAP, D), not the symbolic
 * slice `cache[..., :start_pos+T, :]`. tinygrad cannot build a placeholder for a symbolic extent
 * ("no placeholder-like on symbolic shape"), and the cache buffer is concrete, so the kernel takes
 * the buffer plus `start_pos` as a runtime int and indexes by absolute position.
 *
 *   scores[j] = (q_i . k_j) * (1/sqrt(D))   for j <= pos+i        (causal, absolute positions)
 *   p         = softmax(scores)
 *   out_i     = sum_j p[j] * v_j
 *
 * -nostdlib, so exp comes from a polynomial exp2 (the same coefficients the reference uses).
 * Entry follows the CPU ABI: one packed argument buffer, entry the only non-static function.
 */
#include <stddef.h>

typedef float F4 __attribute__((vector_size(16), aligned(1)));

#define EXP2_POLY(s) ((((((0.00015359208919107914f*(s)+0.0013392627006396651f)*(s) \
  +0.009618384763598442f)*(s)+0.055503472685813904f)*(s)+0.24022644758224487f)*(s) \
  +0.6931471824645996f)*(s)+1.0f)

static inline float exp2_scalar(float x) {
  float q = x < 0.0f ? x - 0.5f : x + 0.5f;
  int qi = (int)q;
  float s = x - (float)qi;
  union { float f; int i; } u;
  u.f = EXP2_POLY(s);
  u.i += qi * 0x800000;
  return u.f;
}

#ifndef TOK
#define TOK 128
#endif
#ifndef CAP
#define CAP 512
#endif

#define H_ 16
#define KVH_ 4
#define D_ 256

static void attn_row(float *out, const float *q, const float *k, const float *v,
                     int i, int pos, int bh_q, int bh_kv) {
  float scores[CAP];
  const int last = pos + i;                     /* absolute index of the last visible key */
  const float inv = 1.0f / 16.0f;               /* 1/sqrt(256) */
  const float *qi = q + ((size_t)bh_q * TOK + i) * D_;
  for (int j = 0; j <= last; j++) {
    const float *kj = k + ((size_t)bh_kv * CAP + j) * D_;
    F4 acc = (F4){0, 0, 0, 0};
    for (int d = 0; d < D_; d += 4) acc = acc + *(const F4 *)(qi + d) * *(const F4 *)(kj + d);
    scores[j] = (acc[0] + acc[1] + acc[2] + acc[3]) * inv;
  }
  float mx = scores[0];
  for (int j = 1; j <= last; j++) if (scores[j] > mx) mx = scores[j];
  const float log2e = 1.4426950216293335f;
  float sum = 0.0f;
  for (int j = 0; j <= last; j++) { scores[j] = exp2_scalar((scores[j] - mx) * log2e); sum += scores[j]; }
  const float inv_sum = 1.0f / sum;
  for (int j = 0; j <= last; j++) scores[j] *= inv_sum;
  float *oi = out + ((size_t)bh_q * TOK + i) * D_;
  for (int d = 0; d < D_; d += 4) *(F4 *)(oi + d) = (F4){0, 0, 0, 0};
  for (int j = 0; j <= last; j++) {
    const float p = scores[j];
    const float *vj = v + ((size_t)bh_kv * CAP + j) * D_;
    const F4 pv = (F4){p, p, p, p};
    for (int d = 0; d < D_; d += 4) *(F4 *)(oi + d) = *(F4 *)(oi + d) + pv * *(const F4 *)(vj + d);
  }
}

void rockchip_attn(void *args) {
  float *out = *(float **)args;
  const float *q = *((const float **)args + 1);
  const float *kv = *((const float **)args + 2);
  const int pos = (int)(*(*((const float **)args + 3)));
  const float *k = kv;
  const float *v = kv + (size_t)KVH_ * CAP * D_;
  for (int h = 0; h < H_; h++) {
    const int bh_q = h, bh_kv = h / (H_ / KVH_);
    for (int i = 0; i < TOK; i++) attn_row(out, q, k, v, i, pos, bh_q, bh_kv);
  }
}

#ifdef PROBE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(int argc, char **argv) {
  const int pos = argc > 1 ? atoi(argv[1]) : 0;
  size_t nq = (size_t)H_ * TOK * D_, nk = (size_t)2 * KVH_ * CAP * D_;
  float *q = malloc(nq * 4), *kv = malloc(nk * 4), *out = malloc(nq * 4);
  srand(7);
  for (size_t i = 0; i < nq; i++) q[i] = (float)rand() / RAND_MAX - 0.5f;
  for (size_t i = 0; i < nk; i++) kv[i] = (float)rand() / RAND_MAX - 0.5f;
  const float *k = kv, *v = kv + (size_t)KVH_ * CAP * D_;
  int posbuf = pos;
  void *args[4] = {out, q, (void *)kv, &posbuf};
  rockchip_attn(args);

  /* double reference for a sample of rows */
  double worst = 0.0;
  for (int h = 0; h < H_; h += 5) {
    int bh_q = h, bh_kv = h / (H_ / KVH_);
    for (int i = 0; i < TOK; i += (TOK > 1 ? 31 : 1)) {
      int last = pos + i;
      double sc[CAP], mx = -1e30, sum = 0.0;
      for (int j = 0; j <= last; j++) {
        double a = 0.0;
        const float *qi = q + (size_t)(bh_q * TOK + i) * D_, *kj = k + (size_t)(bh_kv * CAP + j) * D_;
        for (int d = 0; d < D_; d++) a += (double)qi[d] * kj[d];
        sc[j] = a / 16.0;
        if (sc[j] > mx) mx = sc[j];
      }
      for (int j = 0; j <= last; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
      for (int j = 0; j <= last; j++) sc[j] /= sum;
      for (int d = 0; d < D_; d += 37) {
        double a = 0.0;
        for (int j = 0; j <= last; j++) a += sc[j] * (double)v[(size_t)(bh_kv * CAP + j) * D_ + d];
        double got = out[(size_t)(bh_q * TOK + i) * D_ + d];
        double r = fabs(a - got) / (fabs(a) + 1e-6);
        if (r > worst) worst = r;
      }
    }
  }
  struct timespec a0, b0; double best = 1e9;
  for (int r = 0; r < 3; r++) {
    clock_gettime(CLOCK_MONOTONIC, &a0); rockchip_attn(args); clock_gettime(CLOCK_MONOTONIC, &b0);
    double ms = ((b0.tv_sec - a0.tv_sec) * 1e3 + (b0.tv_nsec - a0.tv_nsec) / 1e6);
    if (ms < best) best = ms;
  }
  /* FLOP count: sum over heads and rows of (last+1) x 2 x D x 2 matmuls */
  double flop = 0.0;
  for (int i = 0; i < TOK; i++) flop += 4.0 * (double)D_ * (pos + i + 1);
  flop *= H_;
  printf("PROBE TOK=%d CAP=%d pos=%d  max rel err vs fp64 = %.3e   %.2f ms  %.2f GFLOP/s\n",
         TOK, CAP, pos, worst, best, flop / (best * 1e6));
  return 0;
}
#endif
