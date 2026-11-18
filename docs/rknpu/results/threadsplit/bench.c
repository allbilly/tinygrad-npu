// threadsplit/bench.c -- measure whether the guarded scalar exp2 used by this
// project's CPU kernels can be parallelised 4-way, and what the reference's
// guard restructuring is worth.
//
// No Python, no tinygrad, no NPU.  Pure C + pthreads.
//
// Variants (all over N contiguous floats, y = x*sigmoid(x) unless stated):
//   V1   guarded scalar, transcribed from the emitted C of E_393216_3
//        (docs/rknpu/results/prof-merged.log), including its 3-elements-per-
//        iteration structure: the emitted kernel handles 3 lanes per loop body,
//        so the compiler already has 3 independent polynomial chains to
//        interleave.  That matters for how much a 4-wide vector can win, so it
//        is part of the transcription rather than flattened away.
//   V1s  same arithmetic, stride-1 simple loop (one chain per element).  Extra
//        row, isolates how much the emitted 3-wide body is worth on its own.
//   V2   V1 with the range split 4 ways over pre-created pthreads.
//   V3   reference vector: float4 ext_vector_type, unguarded polynomial on the
//        fast path, ONE magnitude branch per 4 lanes.
//   V4   V3 with the range split 4 ways over pre-created pthreads.
//   S1   guarded scalar sigmoid (denominator only), single thread.  Extra row.
//   S3   reference vector sigmoid, single thread.  Extra row.
//   COPY y = x, memory floor.
//   V2p, V4p  V2/V3 with 3 pthreads created and joined *per call*, to price the
//        thread-creation cost the pre-created variants avoid.
//   DISP dispatch/barrier only, no work: the floor of the 4-thread path.
//
// Build flags are the project's own, minus the ones that break a hosted
// executable (-nostdlib).  See build.sh.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <stdint.h>

typedef float    float4 __attribute__((ext_vector_type(4)));
typedef int      int4   __attribute__((ext_vector_type(4)));
typedef unsigned uint4  __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------- timing ----
static inline uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void cpu_relax(void) { __asm__ __volatile__("yield" ::: "memory"); }

// ------------------------------------------------- emitted scalar arithmetic -
// Verbatim from prof-merged.log kernel E_393216_3.  Coefficient set and op
// order preserved; only the trailing *gate is dropped (this harness measures
// SiLU, not the gated SwiGLU product).

#define NEG_LOG2E (-1.4426950216293335f)
#define C6 0.00015359208919107914f
#define C5 0.0013392627006396651f
#define C4 0.009618384763598442f
#define C3 0.055503472685813904f
#define C2 0.24022644758224487f
#define C1 0.6931471824645996f

// alu3..alu26 of the emitted kernel, on t = x * -log2(e) already formed.
// Returns the same float the emitted code feeds into (1.0f + alu27).
static inline float exp2_guarded_scalar(float t) {
  _Bool alu4 = (t != t);
  float alu5 = ((t != ((float)(-__builtin_inff()))) ? t : 0.0f);
  float alu6 = (alu4 ? 0.0f : alu5);
  float alu7 = ((t != ((float)(__builtin_inff()))) ? alu6 : 0.0f);
  float alu8 = ((alu7 < 0.0f) ? -0.5f : 0.5f);
  int   cast0 = ((int)((alu7 + alu8)));
  int   alu21 = (cast0 >> 1);
  float alu24 = (alu7 - ((float)(cast0)));
  float alu25 = (((t < 128.0f) == 0) ? ((float)(__builtin_inff()))
               : (((((((((((((C6 * alu24) + C5) * alu24) + C4) * alu24) + C3)
                    * alu24) + C2) * alu24) + C1) * alu24) + 1.0f)
                  * __builtin_bit_cast(float, (int)(((alu21 + 127) << 23)))
                  * __builtin_bit_cast(float, (int)((((cast0 - alu21) + 127) << 23)))));
  float alu26 = ((t < -150.0f) ? 0.0f : alu25);
  return (alu4 ? ((float)(__builtin_nanf(""))) : alu26);
}

// One element, written out as the emitted body does it.
static inline float silu_v1_scalar(float x) {
  float t = x * NEG_LOG2E;
  return x / (1.0f + exp2_guarded_scalar(t));
}
static inline float sigmoid_v1_scalar(float x) {
  float t = x * NEG_LOG2E;
  return 1.0f / (1.0f + exp2_guarded_scalar(t));
}

// ------------------------------------------- reference unguarded vector exp2 -
// Verbatim from the reference's rockchip_exp2_unchecked.  always_inline so the
// hot loop really contains a 4-wide polynomial and not an AAPCS64 call with a
// 4-float stack round-trip per four lanes (measured: not forcing this left V3
// at 3.03 ns/elem instead of the 1.5 ns/elem it reaches inlined, i.e. the
// uninlined form understated the vector win by 2x).
static inline __attribute__((always_inline))
float4 rockchip_exp2_unchecked(float4 x) {
  int4   q = __builtin_convertvector(x + (x < 0 ? (float4)-.5f : (float4).5f), int4);
  float4 s = x - __builtin_convertvector(q, float4);
  float4 y = ((((((C6 * s + C5) * s + C4) * s + C3) * s + C2) * s + C1) * s + 1.0f);
  return __builtin_bit_cast(float4, __builtin_bit_cast(int4, y) + q * (int4)0x800000);
}

// |x| < 125 fast path.  Integer magnitude compare, not fmax: ARM's FMAX/FMAXV
// drop a quiet NaN in favour of the other operand, so a float max could let a
// NaN lane through onto the fast path (where convertvector's out-of-range
// result is undefined).  Integer max over |bits| cannot drop it: the NaN
// encodings all exceed 0x42fa0000, so they take the checked path and reproduce
// V1's NaN exit.  One compare per 4 lanes.
static inline __attribute__((always_inline)) int exp2_fast4(float4 t) {
  uint4 u = __builtin_bit_cast(uint4, t);
  uint4 m = (uint4)(u & (uint4)0x7fffffffu);
  unsigned mm = m[0];
  if (m[1] > mm) mm = m[1];
  if (m[2] > mm) mm = m[2];
  if (m[3] > mm) mm = m[3];
  return mm < 0x42fa0000u;
}

// rare path, out of line so the hot loop stays tight
static __attribute__((noinline)) float4 exp2_slow4(float4 t) {
  float4 r;
  r[0] = exp2_guarded_scalar(t[0]);
  r[1] = exp2_guarded_scalar(t[1]);
  r[2] = exp2_guarded_scalar(t[2]);
  r[3] = exp2_guarded_scalar(t[3]);
  return r;
}

static inline __attribute__((always_inline)) float4 exp2_vec4(float4 t) {
  if (exp2_fast4(t)) return rockchip_exp2_unchecked(t);
  return exp2_slow4(t);
}
static inline __attribute__((always_inline)) float4 silu_v3_vec4(float4 x) {
  float4 t = x * (float4)NEG_LOG2E;
  return x / ((float4)1.0f + exp2_vec4(t));
}
static inline __attribute__((always_inline)) float4 sigmoid_v3_vec4(float4 x) {
  float4 t = x * (float4)NEG_LOG2E;
  return ((float4)1.0f) / ((float4)1.0f + exp2_vec4(t));
}

// ------------------------------------------------------------------- loops ---
enum kind { K_V1 = 0, K_V1S, K_V2, K_V3, K_V4, K_S1, K_S3, K_COPY, K_DISP,
            K_V2P, K_V4P, K_V3E, K_TCREATE, K_V3U2, K_NKIND };

static const char *kname[K_NKIND] = {
  "V1", "V1s", "V2", "V3", "V4", "S1", "S3", "COPY", "DISP", "V2p", "V4p",
  "V3e", "TCREATE", "V3u2"
};

// V1: faithful 3-lanes-per-iteration body.
static void loop_v1(const float *restrict a, float *restrict b, long n) {
  long i = 0;
  for (; i + 3 <= n; i += 3) {
    float val3 = a[i + 1];                    // the emitted body computes lanes
    float val4 = a[i + 2];                    // 1,2,0 of the contiguous triple
    float val5 = a[i + 0];
    float t0 = (val3 * NEG_LOG2E), t1 = (val4 * NEG_LOG2E), t2 = (val5 * NEG_LOG2E);
    float alu27 = exp2_guarded_scalar(t0);
    float alu31 = exp2_guarded_scalar(t1);
    float alu35 = exp2_guarded_scalar(t2);
    b[i + 1] = ((val3 / (1.0f + alu27)));
    b[i + 2] = ((val4 / (1.0f + alu31)));
    b[i + 0] = ((val5 / (1.0f + alu35)));
  }
  for (; i < n; i++) b[i] = silu_v1_scalar(a[i]);
}

// V1s: same arithmetic, stride 1, one chain per element.
static void loop_v1s(const float *restrict a, float *restrict b, long n) {
  for (long i = 0; i < n; i++) b[i] = silu_v1_scalar(a[i]);
}

// V3: 4 lanes per iteration, one guard per 4 lanes.
static void loop_v3(const float *restrict a, float *restrict b, long n) {
  long i = 0;
  for (; i + 4 <= n; i += 4) {
    float4 x = *(const float4 *)(a + i);
    *(float4 *)(b + i) = silu_v3_vec4(x);
  }
  for (; i < n; i++) b[i] = silu_v1_scalar(a[i]);
}

static void loop_s1(const float *restrict a, float *restrict b, long n) {
  for (long i = 0; i < n; i++) b[i] = sigmoid_v1_scalar(a[i]);
}
static void loop_s3(const float *restrict a, float *restrict b, long n) {
  long i = 0;
  for (; i + 4 <= n; i += 4) {
    float4 x = *(const float4 *)(a + i);
    *(float4 *)(b + i) = sigmoid_v3_vec4(x);
  }
  for (; i < n; i++) b[i] = sigmoid_v1_scalar(a[i]);
}
static void loop_copy(const float *restrict a, float *restrict b, long n) {
  for (long i = 0; i < n; i++) b[i] = a[i];
}
// V3e: the vector exp2 with no divide, to price the polynomial+load/store
// alone.  Not a meaningful SiLU: an ablation only.
static void loop_v3e(const float *restrict a, float *restrict b, long n) {
  long i = 0;
  for (; i + 4 <= n; i += 4) {
    float4 x = *(const float4 *)(a + i);
    float4 t = x * (float4)NEG_LOG2E;
    *(float4 *)(b + i) = exp2_vec4(t);
  }
  for (; i < n; i++) b[i] = exp2_guarded_scalar(a[i] * NEG_LOG2E);
}

// ---- extra ablation, NOT part of the reference: the same float4 arithmetic
// with two independent 4-lane chains per iteration and one guard per 8 lanes.
// The single-chain version is latency-bound on six dependent fmla (4 cycles
// each); this shows what the vector form reaches once that chain is broken.
static inline __attribute__((always_inline)) int exp2_fast8(float4 t0, float4 t1) {
  uint4 u0 = (uint4)(__builtin_bit_cast(uint4, t0) & (uint4)0x7fffffffu);
  uint4 u1 = (uint4)(__builtin_bit_cast(uint4, t1) & (uint4)0x7fffffffu);
  unsigned mm = u0[0];
  if (u0[1] > mm) mm = u0[1];
  if (u0[2] > mm) mm = u0[2];
  if (u0[3] > mm) mm = u0[3];
  if (u1[0] > mm) mm = u1[0];
  if (u1[1] > mm) mm = u1[1];
  if (u1[2] > mm) mm = u1[2];
  if (u1[3] > mm) mm = u1[3];
  return mm < 0x42fa0000u;
}
static void loop_v3u2(const float *restrict a, float *restrict b, long n) {
  long i = 0;
  for (; i + 8 <= n; i += 8) {
    float4 x0 = *(const float4 *)(a + i);
    float4 x1 = *(const float4 *)(a + i + 4);
    float4 t0 = x0 * (float4)NEG_LOG2E;
    float4 t1 = x1 * (float4)NEG_LOG2E;
    float4 e0, e1;
    if (exp2_fast8(t0, t1)) {
      e0 = rockchip_exp2_unchecked(t0);
      e1 = rockchip_exp2_unchecked(t1);
    } else {
      e0 = exp2_slow4(t0);
      e1 = exp2_slow4(t1);
    }
    *(float4 *)(b + i)     = x0 / ((float4)1.0f + e0);
    *(float4 *)(b + i + 4) = x1 / ((float4)1.0f + e1);
  }
  for (; i < n; i++) b[i] = silu_v1_scalar(a[i]);
}

// K_V2/V4 dispatch to V1/V3; K_V2P/V4P same body, threads made per call.
static void do_chunk(int kind, const float *a, float *b, long s, long e) {
  long n = e - s;
  switch (kind) {
    case K_V1: case K_V2: case K_V2P: loop_v1(a + s, b + s, n); break;
    case K_V1S: loop_v1s(a + s, b + s, n); break;
    case K_V3: case K_V4: case K_V4P: loop_v3(a + s, b + s, n); break;
    case K_S1: loop_s1(a + s, b + s, n); break;
    case K_S3: loop_s3(a + s, b + s, n); break;
    case K_COPY: loop_copy(a + s, b + s, n); break;
    case K_V3E: loop_v3e(a + s, b + s, n); break;
    case K_V3U2: loop_v3u2(a + s, b + s, n); break;
    case K_DISP: break;
    default: break;
  }
}

// ------------------------------------------------------- persistent threads -
#define NW 4
static volatile long g_go, g_done;
static volatile int  g_stop;
static int           g_kind;
static const float  *g_a;
static float        *g_b;
static long          g_start[NW], g_end[NW];

static void *worker(void *arg) {
  int id = (int)(intptr_t)arg;
  long last = 0;
  for (;;) {
    long my;
    while ((my = __atomic_load_n(&g_go, __ATOMIC_ACQUIRE)) == last) {
      if (__atomic_load_n(&g_stop, __ATOMIC_RELAXED)) return NULL;
      cpu_relax();
    }
    last = my;
    do_chunk(g_kind, g_a, g_b, g_start[id], g_end[id]);
    __atomic_fetch_add(&g_done, 1, __ATOMIC_ACQ_REL);
  }
  return NULL;
}

static pthread_t g_th[NW - 1];   // thread 0 is the caller

static void pin(int cpu) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  if (sched_setaffinity(0, sizeof(s), &s) != 0) perror("sched_setaffinity");
}

// Split into NW ranges.  thread 0 == caller.
static void split(long n) {
  long chunk = (n / NW) & ~15L;       // 64 B aligned boundaries
  for (int i = 0; i < NW; i++) {
    g_start[i] = (long)i * chunk;
    g_end[i]   = (i == NW - 1) ? n : (long)(i + 1) * chunk;
  }
}

// caller runs chunk 0 itself; workers were pinned inside worker_pinned

// workers need their own affinity; do it from the worker's own context
static void *worker_pinned(void *arg) {
  int id = (int)(intptr_t)arg;
  pin(4 + id);
  return worker(arg);
}

// per-call thread struct
typedef struct { int id; int kind; const float *a; float *b; long s, e; } job_t;
static void *job_run(void *p) {
  job_t *j = (job_t *)p;
  pin(4 + j->id);
  do_chunk(j->kind, j->a, j->b, j->s, j->e);
  return NULL;
}

static void par_run(int kind, const float *a, float *b, long n) {
  g_kind = kind; g_a = a; g_b = b;
  split(n);
  __atomic_store_n(&g_done, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&g_go, __atomic_load_n(&g_go, __ATOMIC_RELAXED) + 1, __ATOMIC_RELEASE);
  do_chunk(kind, a, b, g_start[0], g_end[0]);
  while (__atomic_load_n(&g_done, __ATOMIC_ACQUIRE) < NW - 1) cpu_relax();
}

static void par_run_percall(int kind, const float *a, float *b, long n) {
  split(n);
  pthread_t th[NW - 1];
  job_t j[NW - 1];
  for (int i = 1; i < NW; i++) {
    j[i - 1] = (job_t){ i, kind, a, b, g_start[i], g_end[i] };
    if (pthread_create(&th[i - 1], NULL, job_run, &j[i - 1]) != 0) { perror("pthread_create"); exit(1); }
  }
  do_chunk(kind, a, b, g_start[0], g_end[0]);
  for (int i = 1; i < NW; i++) pthread_join(th[i - 1], NULL);
}

static void run_once(int kind, const float *a, float *b, long n) {
  switch (kind) {
    case K_V1: case K_V1S: case K_V3: case K_S1: case K_S3: case K_COPY:
    case K_V3E: case K_V3U2:
      do_chunk(kind, a, b, 0, n); break;
    case K_DISP: par_run(K_DISP, a, b, n); break;
    case K_TCREATE: par_run_percall(K_DISP, a, b, n); break;   // 3 threads, no work
    case K_V2P: case K_V4P: par_run_percall(kind, a, b, n); break;
    default: par_run(kind, a, b, n); break;
  }
}

// ------------------------------------------------------------- correctness --
// bit-identity / ULP between two same-length float arrays
static int32_t fbits(float f) { int32_t v; memcpy(&v, &f, 4); return v; }
static int32_t ordered(float f) {
  int32_t v = fbits(f);
  return v < 0 ? (int32_t)(0x80000000u - (uint32_t)v) : v;
}
static long compare(const float *ref, const float *got, long n, const char *label,
                    int *maxulp_out, long *first_out) {
  long diff = 0; int maxulp = 0; long first = -1;
  for (long i = 0; i < n; i++) {
    int32_t d = ordered(ref[i]) - ordered(got[i]);
    if (d < 0) d = -d;
    if (d) {
      diff++;
      if (first < 0) first = i;
      if (d > maxulp) maxulp = d;
    }
  }
  *maxulp_out = maxulp; *first_out = first;
  printf("# cmp %-28s n=%ld differing=%ld max_ulp=%d first_mismatch=%ld\n",
         label, n, diff, maxulp, first);
  return diff;
}

// ------------------------------------------------------------------- main ----
// ---- buffers shared by the sweep driver
static float *gA, *gB, *REF, *TMP;

// ---- start/stop the persistent worker pool.  Between phases the threads are
// joined, so a phase that does not need them runs with no unrelated spinning
// thread on cpu5-7.  (An earlier version of this harness left them spinning and
// starved freshly created threads for ~4.4 ms each -- CFS timeslice, not
// thread-create cost; measured separately and discarded.  See report.md.)
static int g_running;
static void workers_start(void) {
  if (g_running) return;
  __atomic_store_n(&g_stop, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&g_go, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&g_done, 0, __ATOMIC_RELAXED);
  for (int i = 1; i < NW; i++)
    if (pthread_create(&g_th[i - 1], NULL, worker_pinned, (void *)(intptr_t)i) != 0) {
      perror("pthread_create"); exit(1);
    }
  g_running = 1;
}
static void workers_stop(void) {
  if (!g_running) return;
  __atomic_store_n(&g_stop, 1, __ATOMIC_RELEASE);
  for (int i = 1; i < NW; i++) pthread_join(g_th[i - 1], NULL);
  g_running = 0;
}

static int  g_reps;

// Measure one interleaved sweep: for each round, run every variant in order and
// print every individual rep.  Nothing is averaged; the median is taken by the
// post-processor over the printed raw values.
static void sweep(const int *order, int nk, long n, int reps, const char *phase) {
  printf("# phase %s n=%ld reps=%d\n", phase, n, reps);
  for (int k = 0; k < nk; k++)
    for (int w = 0; w < 3; w++) run_once(order[k], gA, gB, n);   // warm-up
  for (int r = 0; r < reps; r++)
    for (int k = 0; k < nk; k++) {
      int kind = order[k];
      uint64_t t0 = now_ns();
      run_once(kind, gA, gB, n);
      uint64_t t1 = now_ns();
      double ns = (double)(t1 - t0);
      printf("%s,%ld,%d,%.0f,%.4f,%.3f\n", kname[kind], n, r, ns, ns / 1e6, ns / (double)n);
    }
  fflush(stdout);
}

int main(int argc, char **argv) {
  int reps = (argc > 1) ? atoi(argv[1]) : 20;
  int reps_extra = (argc > 2) ? atoi(argv[2]) : 10;
  int only = (argc > 3) ? atoi(argv[3]) : 0;   // 0 = all phases, 3 = per-call only
  long sizes[2] = { 131072, 1179648 };

  gA = aligned_alloc(64, (size_t)1179648 * 4);
  gB = aligned_alloc(64, (size_t)1179648 * 4);
  REF = aligned_alloc(64, (size_t)1179648 * 4);
  TMP = aligned_alloc(64, (size_t)1179648 * 4);

  pin(4);   // caller stays on cpu4 (Cortex-A76) for the whole run

  printf("# threadsplit bench: %d reps (single/multi), %d reps (extras/per-call)\n",
         reps, reps_extra);
  printf("# affinity: caller/thread0 = cpu4, workers = cpu5,6,7 (all Cortex-A76);\n");
  printf("#           no thread ever runs on cpu0-3 (Cortex-A55).\n");
  printf("# columns: variant,size,rep,ns,ms,ns_per_elem\n");

  for (int si = 0; si < 2; si++) {
    long n = sizes[si];
    for (long i = 0; i < n; i++) gA[i] = (float)((i % 2001) - 1000) * 0.02f;  // |x| <= 20
    memset(gB, 0, (size_t)n * 4);
    g_reps = reps;

    if (only == 0 || only == 1) {
      // ---- phase 1: no worker threads exist at all -> quiet machine.
      int p1[] = { K_V1, K_V1S, K_V3, K_V3U2, K_S1, K_S3, K_V3E, K_COPY };
      sweep(p1, (int)(sizeof p1 / sizeof p1[0]), n, reps, "1-single");
    }

    // ---- phase 2: persistent workers live and spin.  V1/V3 are repeated here
    // as in-phase controls, so V2/V1 and V4/V3 are ratios of two measurements
    // taken under identical conditions.
    if (only == 0 || only == 2) {
      workers_start();
      int p2[] = { K_V1, K_V2, K_V3, K_V4, K_DISP };
      sweep(p2, (int)(sizeof p2 / sizeof p2[0]), n, reps, "2-parallel");
      workers_stop();
    }

    if (only == 0 || only == 3) {
      // ---- phase 3: no threads alive; create 3 per call and join.
      int p3[] = { K_TCREATE, K_V2P, K_V4P };
      sweep(p3, (int)(sizeof p3 / sizeof p3[0]), n, reps_extra, "3-percall");
    }
  }

  // ------------------------------------------------------- correctness ------
  if (only) { printf("# phase-only run %d: skipping correctness\n", only); return 0; }
  printf("\n# ---- correctness ----\n");
  workers_start();
  for (int si = 0; si < 2; si++) {
    long n = sizes[si];
    int maxulp; long first; char lbl[160];

    for (long i = 0; i < n; i++) gA[i] = -20.0f + 40.0f * (float)i / (float)(n - 1);
    do_chunk(K_V1, gA, REF, 0, n);
    do_chunk(K_V3, gA, TMP, 0, n);
    snprintf(lbl, sizeof lbl, "V3 vs V1  x in [-20,20]");
    compare(REF, TMP, n, lbl, &maxulp, &first);
    par_run(K_V4, gA, TMP, n);
    snprintf(lbl, sizeof lbl, "V4 vs V1  x in [-20,20]");
    compare(REF, TMP, n, lbl, &maxulp, &first);
    par_run(K_V2, gA, TMP, n);
    snprintf(lbl, sizeof lbl, "V2 vs V1  x in [-20,20]");
    compare(REF, TMP, n, lbl, &maxulp, &first);
    do_chunk(K_V1S, gA, TMP, 0, n);
    snprintf(lbl, sizeof lbl, "V1s vs V1 x in [-20,20]");
    compare(REF, TMP, n, lbl, &maxulp, &first);
    do_chunk(K_S3, gA, TMP, 0, n);
    do_chunk(K_S1, gA, REF, 0, n);
    snprintf(lbl, sizeof lbl, "S3 vs S1  x in [-20,20]");
    compare(REF, TMP, n, lbl, &maxulp, &first);

    // wide sweep: crosses the slow-path guard at |t| = 125
    for (long i = 0; i < n; i++) gA[i] = -1000.0f + 2000.0f * (float)i / (float)(n - 1);
    do_chunk(K_V1, gA, REF, 0, n);
    do_chunk(K_V3, gA, TMP, 0, n);
    snprintf(lbl, sizeof lbl, "V3 vs V1  x in [-1000,1000]");
    long d = compare(REF, TMP, n, lbl, &maxulp, &first);
    if (d) {
      long shown = 0;
      for (long i = 0; i < n && shown < 6; i++)
        if (ordered(REF[i]) != ordered(TMP[i])) {
          printf("#   x=%.9g  V1=%.9g (%08x)  V3=%.9g (%08x)\n", gA[i], REF[i],
                 (unsigned)fbits(REF[i]), TMP[i], (unsigned)fbits(TMP[i]));
          shown++;
        }
    }
    par_run(K_V4, gA, TMP, n);
    snprintf(lbl, sizeof lbl, "V4 vs V1  x in [-1000,1000]");
    compare(REF, TMP, n, lbl, &maxulp, &first);

    // guard boundary probes
    float probe[] = { -1000.f, -200.f, -130.f, -90.f, -86.7f, -86.6f, -50.f, -20.f,
                      -1.f, 0.f, 1.f, 20.f, 50.f, 86.6f, 86.7f, 90.f, 120.f,
                      125.f, 127.f, 128.f, 130.f, 200.f, 1000.f };
    int np = (int)(sizeof probe / sizeof probe[0]);
    for (int i = 0; i < np; i++) gA[i] = probe[i];
    for (long i = np; i < n; i++) gA[i] = 0.f;
    do_chunk(K_V1, gA, REF, 0, np);
    do_chunk(K_V3, gA, TMP, 0, np);
    printf("# guard probe N=%ld (V1 scalar vs V3 vector):\n", n);
    int bad = 0;
    for (int i = 0; i < np; i++) {
      int same = (fbits(REF[i]) == fbits(TMP[i]));
      if (!same) bad++;
      printf("#   x=%-8g V1=%-14.9g V3=%-14.9g %s\n", probe[i], REF[i], TMP[i],
             same ? "same" : "DIFF");
    }
    printf("#   -> %d/%d differ at the guard probes\n", bad, np);

    float sp[] = { NAN, INFINITY, -INFINITY, 0.0f, -0.0f, 1e-30f, -1e-30f };
    int nsp = (int)(sizeof sp / sizeof sp[0]);
    for (int i = 0; i < nsp; i++) gA[i] = sp[i];
    for (long i = nsp; i < n; i++) gA[i] = 0.f;
    do_chunk(K_V1, gA, REF, 0, nsp);
    do_chunk(K_V3, gA, TMP, 0, nsp);
    printf("# specials (V1 vs V3):\n");
    for (int i = 0; i < nsp; i++)
      printf("#   x=%-10g V1=%-14.9g V3=%-14.9g %s\n", sp[i], REF[i], TMP[i],
             fbits(REF[i]) == fbits(TMP[i]) ? "same" : "DIFF");
  }
  workers_stop();
  return 0;
}
