/* Bitwise A/B for a loop-split CPU kernel.
 *
 * Compiles the original emitted kernel and the thread_split-transformed one into the same
 * binary under different names, feeds both identically-filled buffers, and compares the
 * output bytes. A loop split is exact when the loop carries no state across iterations, so
 * the expected result is memcmp == 0. Any difference means the chosen loop was wrong.
 *
 * Built by the sibling run.sh; not part of the model path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void kernel_orig(unsigned char *);
void kernel_split(unsigned char *);

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 393216;
  size_t out_bytes = (size_t)n * 3 * sizeof(float) + 64;
  size_t in_bytes = (size_t)n * 3 * sizeof(float) + 64;
  float *o0 = aligned_alloc(64, out_bytes), *o1 = aligned_alloc(64, out_bytes);
  float *i0 = aligned_alloc(64, in_bytes), *i1 = aligned_alloc(64, in_bytes);
  float *j0 = aligned_alloc(64, in_bytes), *j1 = aligned_alloc(64, in_bytes);
  memset(o0, 0, out_bytes); memset(o1, 0, out_bytes);
  srand(1234);
  for (size_t k = 0; k < (size_t)n * 3; k++) {
    /* same values in both copies; wide enough to exercise the exp2 guards */
    float v = ((float)rand() / (float)RAND_MAX - 0.5f) * 60.0f;
    i0[k] = i1[k] = v;
    float u = ((float)rand() / (float)RAND_MAX - 0.5f) * 8.0f;
    j0[k] = j1[k] = u;
  }
  unsigned long a0[3] = {(unsigned long)o0, (unsigned long)i0, (unsigned long)j0};
  unsigned long a1[3] = {(unsigned long)o1, (unsigned long)i1, (unsigned long)j1};
  kernel_orig((unsigned char *)a0);
  kernel_split((unsigned char *)a1);
  int bad = 0;
  for (size_t k = 0; k < (size_t)n * 3; k++) {
    if (memcmp(&o0[k], &o1[k], sizeof(float))) {
      if (bad < 5) printf("diff at %zu: %.9g vs %.9g\n", k, o0[k], o1[k]);
      bad++;
    }
  }
  printf("n=%d elements=%zu differing=%d verdict=%s\n", n, (size_t)n * 3, bad, bad ? "FAIL" : "BITWISE-EQUAL");
  if (argc > 2) {
    /* interleaved A/B: alternate the two variants so thermal drift hits both equally */
    int reps = atoi(argv[2]);
    double t_orig = 0, t_split = 0;
    for (int r = 0; r < reps; r++) {
      struct timespec a, b;
      clock_gettime(CLOCK_MONOTONIC, &a);
      kernel_orig((unsigned char *)a0);
      clock_gettime(CLOCK_MONOTONIC, &b);
      t_orig += (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
      clock_gettime(CLOCK_MONOTONIC, &a);
      kernel_split((unsigned char *)a1);
      clock_gettime(CLOCK_MONOTONIC, &b);
      t_split += (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
    }
    printf("reps=%d  serial %.2f ms (%.3f ms/call)  split %.2f ms (%.3f ms/call)  speedup %.2fx\n",
           reps, t_orig, t_orig / reps, t_split, t_split / reps, t_orig / t_split);
  }
  return bad ? 1 : 0;
}
