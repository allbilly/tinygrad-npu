/* Does the GDN scan actually scale four ways, with the real kernel and the real pool?
 *
 * Our `recurrent.c` is byte-identical to the reference's, the call count is the same (24), yet
 * theirs measures 7.39 ms/call against our 25.1, and our pooled dispatch measured only 1.17x.
 * That is the single largest item in the remaining gap (~425 ms), so this harness settles it
 * without a model run:
 *
 *   serial   cpu_gated_delta_scan(...)
 *   pooled   cpu_gated_delta_scan_pooled(..., (scan_parallel_fn)rk_run, &status)
 *
 * plus a 4-way pthread-per-call variant, to show the cost that made the first attempt fail.
 * Outputs are compared byte for byte: a head partition must not change any value.
 *
 * Build (from this directory):
 *   clang -O2 -o scanprobe scanprobe.c ../../recurrent.c ../rkpool.c -lpthread -I.
 * Run pinned to the four A76s:
 *   taskset -c 4-7 ./scanprobe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int (*scan_parallel_fn)(void (*)(void **, int, int), void **, int);
extern int cpu_gated_delta_scan(float *, const float *, const float *, const float *,
                                const float *, const float *, float *, int, int, int, int,
                                int, int, int, int, float *);
extern int cpu_gated_delta_scan_pooled(float *, const float *, const float *, const float *,
                                       const float *, const float *, float *, int, int, int, int,
                                       int, int, int, int, float *, scan_parallel_fn, int *);
extern int rk_run(void (*)(void **, int, int), void **, int);
extern int cpu_gated_delta_scan_part(float *, const float *, const float *, const float *,
                                     const float *, const float *, float *, int, int, int, int,
                                     int, int, int, int, float *, int, int);

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
  const int B = 1, H = 32, T = 128, K = 128, V = 128, A = 1;
  const size_t nq = (size_t)B * H * T * K, navec = (size_t)B * H * T * A;
  const size_t nstate = (size_t)B * H * V * K, nout = (size_t)B * H * T * V;
  int reps = argc > 1 ? atoi(argv[1]) : 5;

  float *q = malloc(nq * 4), *k = malloc(nq * 4), *v = malloc(nq * 4);
  float *beta = malloc(navec * 4), *alpha = malloc(navec * 4);
  float *st0 = malloc(nstate * 4), *st1 = malloc(nstate * 4);
  float *o0 = malloc(nout * 4), *o1 = malloc(nout * 4);
  srand(7);
  for (size_t i = 0; i < nq; i++) { q[i] = (rand()/(float)RAND_MAX-0.5f); k[i] = (rand()/(float)RAND_MAX-0.5f); v[i] = (rand()/(float)RAND_MAX-0.5f); }
  for (size_t i = 0; i < navec; i++) { beta[i] = 0.5f; alpha[i] = 0.9f; }
  for (size_t i = 0; i < nstate; i++) { st0[i] = (rand()/(float)RAND_MAX-0.5f)*0.1f; st1[i] = st0[i]; }
  memset(o0, 0, nout * 4); memset(o1, 0, nout * 4);

  int status = -1;
  int rc = cpu_gated_delta_scan_pooled(o1, q, k, v, beta, alpha, st1, 0, T, B, H, T, K, V, A, NULL,
                                       (scan_parallel_fn)rk_run, &status);
  cpu_gated_delta_scan(o0, q, k, v, beta, alpha, st0, 0, T, B, H, T, K, V, A, NULL);
  int diff = 0;
  for (size_t i = 0; i < nout; i++) if (memcmp(&o0[i], &o1[i], 4)) diff++;
  for (size_t i = 0; i < nstate; i++) if (memcmp(&st0[i], &st1[i], 4)) diff++;
  printf("status=%d rc=%d differing=%d verdict=%s\n", status, rc, diff, diff ? "FAIL" : "BITWISE-EQUAL");

  double ts = 0, tp = 0;
  for (int r = 0; r < reps; r++) {
    double a = now_ms();
    cpu_gated_delta_scan(o0, q, k, v, beta, alpha, st0, 0, T, B, H, T, K, V, A, NULL);
    double b = now_ms();
    cpu_gated_delta_scan_pooled(o1, q, k, v, beta, alpha, st1, 0, T, B, H, T, K, V, A, NULL,
                                (scan_parallel_fn)rk_run, &status);
    double c = now_ms();
    ts += b - a; tp += c - b;
  }
  printf("reps=%d  serial %.2f ms/call  pooled %.2f ms/call  speedup %.2fx  status=%d\n",
         reps, ts/reps, tp/reps, ts/tp, status);
  /* One quarter of the work, alone: if this is ~pooled, the pool is running the groups
     essentially serially; if it is ~pooled/1, the groups overlap. */
  double g = 0;
  for (int r = 0; r < reps; r++) {
    double a = now_ms();
    cpu_gated_delta_scan_part(o0, q, k, v, beta, alpha, st0, 0, T, B, H, T, K, V, A, NULL, 0, 4);
    g += now_ms() - a;
  }
  printf("one group (of 4) alone: %.2f ms/call  |  pooled/group = %.2f\n", g/reps, (tp/reps)/(g/reps));
  return diff ? 1 : 0;
}
