/* Function-boundary bitwise parity for the four-way split.
 *
 * Both kernels receive eight identical 32 MB buffers, filled from a fixed pseudo-random stream,
 * and the *entire* memory is compared afterwards. Comparing everything (rather than the declared
 * outputs) means the driver needs no knowledge of each kernel's argument sizes, and any byte a
 * kernel writes outside its intended range also shows up as a difference.
 *
 * Linked against librkpool.so by symbol, which is the path already proven bitwise-equal for
 * E_2_16_32_32_4_4. Usage: ./parity4 <orig_fn> <split_fn>  (both take unsigned char *args)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void kern_t(unsigned char *);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

#define NARG 8
#define SZ   (32u << 20)

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <orig_fn> <split_fn> [reps]\n", argv[0]); return 2; }
  kern_t *orig = (kern_t *)strtoull(argv[1], 0, 16);
  kern_t *split = (kern_t *)strtoull(argv[2], 0, 16);
  int reps = argc > 3 ? atoi(argv[3]) : 5;

  unsigned char *a = malloc((size_t)NARG * SZ), *b = malloc((size_t)NARG * SZ);
  if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 2; }
  /* finite inputs only: buffers 4/5 are beta/alpha, the rest are activations */
  unsigned int seed = 12345;
  for (size_t buf = 0; buf < NARG; buf++) {
    float *f = (float *)(a + buf * SZ);
    size_t n = SZ / sizeof(float);
    if (buf == 4) { for (size_t i = 0; i < n; i++) f[i] = 0.5f; continue; }
    if (buf == 5) { for (size_t i = 0; i < n; i++) f[i] = 0.9f; continue; }
    for (size_t i = 0; i < n; i++) {
      seed = seed * 1664525u + 1013904223u;
      f[i] = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 0.4f;
    }
  }
  memcpy(b, a, (size_t)NARG * SZ);

  unsigned long pa[NARG], pb[NARG];
  for (int i = 0; i < NARG; i++) { pa[i] = (unsigned long)(a + (size_t)i * SZ); pb[i] = (unsigned long)(b + (size_t)i * SZ); }

  orig((unsigned char *)pa);
  split((unsigned char *)pb);

  size_t first_diff = (size_t)-1; int ndiff = 0;
  for (size_t i = 0; i < (size_t)NARG * SZ; i++) if (a[i] != b[i]) { if (first_diff == (size_t)-1) first_diff = i; ndiff++; }
  printf("differing bytes: %d", ndiff);
  if (ndiff) printf("  first at offset %zu (buffer %zu, +%zu)", first_diff, first_diff / SZ, first_diff % SZ);
  printf("   verdict=%s\n", ndiff ? "FAIL" : "BITWISE-EQUAL");
  if (ndiff) return 1;

  double ts = 0, tp = 0;
  for (int r = 0; r < reps; r++) {
    double t0 = now(); orig((unsigned char *)pa); double t1 = now();
    split((unsigned char *)pb); double t2 = now();
    ts += t1 - t0; tp += t2 - t1;
  }
  printf("reps=%d  serial %.2f ms  split %.2f ms  speedup %.2fx\n", reps, ts/reps, tp/reps, ts/tp);
  return 0;
}
