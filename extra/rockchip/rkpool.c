/* Four-way work pool, built as a normal shared object.
 *
 * Why a separate .so: kernels compiled by tinygrad's ClangCompiler are loaded with
 * `jit_loader`, which gives file-scope data address 0 on this tree (measured: `&g_x == 0x0`,
 * and stores to it are lost). Statics inside a dlopen'd .so are real, so the pool's state
 * lives here, and the kernel receives rk_run's address as an integer baked into its source --
 * exactly how the reference passes its dispatcher into `rockchip_norm_quantize`.
 *
 * Contract: rk_run(fn, args, groups) calls fn(args, 0, g) for g = 0..split-1, running
 * split-1 of them on parked workers and one on the caller. Returns 0 on success; a nonzero
 * return means the pool could not start and the caller must run every group serially.
 *
 * fn receives (args, unused, group); args is an opaque pointer the kernel builds locally.
 */
typedef unsigned long rk_thread_t;
extern int pthread_create(rk_thread_t *, const void *, void *(*)(void *), void *);
extern int sched_setaffinity(int, unsigned long, const void *);

typedef struct { unsigned long bits[16]; } rk_cpuset;
typedef void (*rk_fn)(void **, int, int);

#define RK_MAX 4
#define RK_WORKERS (RK_MAX - 1)

static struct {
  volatile int seq;      /* incremented by the caller to publish a task */
  volatile int done;     /* incremented by each worker when its group finishes */
  rk_fn fn;
  void **args;
  int group;
} rk_slot[RK_WORKERS];

static rk_thread_t rk_tid[RK_WORKERS];
/* Interval recording, for measuring achieved overlap rather than assuming it. */
extern int clock_gettime(int, void *);
static double rk_t0;
static double rk_start[RK_MAX], rk_end[RK_MAX];
static int rk_cpu[RK_MAX];
extern int sched_getcpu(void);
int rk_group_cpu(int i) { return i >= 0 && i < RK_MAX ? rk_cpu[i] : -1; }
static double rk_now(void) { struct { long s, ns; } t; clock_gettime(1, &t); return t.s * 1e3 + t.ns / 1e6; }
double rk_group_start(int i) { return i >= 0 && i < RK_MAX ? rk_start[i] - rk_t0 : 0; }
double rk_group_end(int i) { return i >= 0 && i < RK_MAX ? rk_end[i] - rk_t0 : 0; }
static volatile int rk_started;
static volatile int rk_failed;
/* Accounting: how often the pool was asked, and how often it declined (caller ran serial). */
static long rk_calls_total, rk_refused_total;
extern long write(int, const void *, unsigned long);
long rk_calls(void) { return rk_calls_total; }
long rk_refused(void) { return rk_refused_total; }

static void *rk_worker(void *opaque) {
  int idx = (int)(long)opaque;
  /* One distinct core per worker. Allowing every worker on all of 4-7 lets the scheduler
     stack two groups on one core and leave another idle -- measured: two groups on cpu7,
     cpu6 unused, which halves the achieved overlap. The caller keeps 4-7 to itself, so it
     lands on whichever of the four the three workers did not take. */
  rk_cpuset set;
  for (int i = 0; i < 16; i++) set.bits[i] = 0;
  set.bits[0] = 1UL << (4 + idx);
  sched_setaffinity(0, sizeof(set), &set);
  int seen = 0;
  for (;;) {
    /* Bounded spin then wfe: wfe keeps three cores from burning power between kernels
       (the NPU ioctl blocks for hundreds of ms), while the spin keeps the wake latency low
       for back-to-back calls. */
    int spins = 0;
    while (__atomic_load_n(&rk_slot[idx].seq, __ATOMIC_ACQUIRE) == seen) {
      if (++spins > 2000) __asm__ volatile("wfe" ::: "memory");
    }
    seen = __atomic_load_n(&rk_slot[idx].seq, __ATOMIC_ACQUIRE);
    rk_cpu[rk_slot[idx].group] = sched_getcpu();
    rk_start[rk_slot[idx].group] = rk_now();
    rk_slot[idx].fn(rk_slot[idx].args, 0, rk_slot[idx].group);
    rk_end[rk_slot[idx].group] = rk_now();
    __atomic_fetch_add(&rk_slot[idx].done, 1, __ATOMIC_RELEASE);
  }
}

int rk_run(rk_fn fn, void **args, int groups) {
  if (groups == 0) groups = RK_MAX;   /* the reference's scan passes 0 for "auto" */
  rk_calls_total++;
  rk_t0 = rk_now();
  if (groups > RK_MAX || groups < 1) return -1;
  if (!rk_started) {
    rk_started = 1;
    for (int i = 0; i < RK_WORKERS; i++)
      if (pthread_create(&rk_tid[i], (void *)0, rk_worker, (void *)(long)i) != 0) rk_failed = 1;
  }
  if (rk_failed) { rk_refused_total++; return -1; }
  int workers = groups - 1;
  for (int i = 0; i < workers; i++) {
    rk_slot[i].done = 0;
    rk_slot[i].fn = fn;
    rk_slot[i].args = args;
    rk_slot[i].group = i;
    __atomic_store_n(&rk_slot[i].seq, rk_slot[i].seq + 1, __ATOMIC_RELEASE);
  }
  __asm__ volatile("sev" ::: "memory");
  rk_cpu[groups - 1] = sched_getcpu();
  rk_start[groups - 1] = rk_now();
  fn(args, 0, groups - 1);                     /* the caller runs the last group */
  rk_end[groups - 1] = rk_now();
  for (int i = 0; i < workers; i++)
    while (__atomic_load_n(&rk_slot[i].done, __ATOMIC_ACQUIRE) == 0) __asm__ volatile("wfe");
  return 0;
}
