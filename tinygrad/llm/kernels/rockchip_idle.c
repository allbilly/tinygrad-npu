#define _GNU_SOURCE
#include <stdatomic.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
static pthread_t thread;
static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition=PTHREAD_COND_INITIALIZER;
static _Atomic int state, busy, ready, tid;
static int started, error;
static void spin(void){__asm__ volatile("yield" ::: "memory");}
static void *run(void *unused){
  (void)unused;
  atomic_store(&tid,(int)syscall(SYS_gettid));
  if(setpriority(PRIO_PROCESS,atomic_load(&tid),19))error=errno;
  atomic_store_explicit(&ready,1,memory_order_release);
  if(error)return 0;
  for(;;){
    pthread_mutex_lock(&mutex);
    while(atomic_load_explicit(&state,memory_order_acquire)==0)pthread_cond_wait(&condition,&mutex);
    int value=atomic_load_explicit(&state,memory_order_acquire);
    if(value==2){pthread_mutex_unlock(&mutex);return 0;}
    atomic_store_explicit(&busy,1,memory_order_release);
    pthread_mutex_unlock(&mutex);
    while(atomic_load_explicit(&state,memory_order_acquire)==1)spin();
    atomic_store_explicit(&busy,0,memory_order_release);
  }
}
int start_spinner(void){
  if(started)return EALREADY;
  atomic_store(&state,0);atomic_store(&busy,0);atomic_store(&ready,0);
  int ret=pthread_create(&thread,0,run,0);
  if(ret)return ret;
  while(!atomic_load_explicit(&ready,memory_order_acquire))spin();
  if(error){pthread_join(thread,0);return error;}
  started=1;return 0;
}
static int activate(int cpu,int wait){
  if(!started||atomic_load(&state)!=0)return EBUSY;
  if(wait)while(atomic_load_explicit(&busy,memory_order_acquire))spin();
  cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);
  int ret=pthread_setaffinity_np(thread,sizeof(mask),&mask);
  if(ret)return ret;
  pthread_mutex_lock(&mutex);atomic_store_explicit(&state,1,memory_order_release);
  pthread_cond_signal(&condition);pthread_mutex_unlock(&mutex);
  if(wait)while(!atomic_load_explicit(&busy,memory_order_acquire))spin();
  return 0;
}
int activate_spinner(int cpu){return activate(cpu,1);}
int activate_spinner_async(int cpu){return activate(cpu,0);}
static int park_spinner(int wait){
  if(!started)return 0;
  pthread_mutex_lock(&mutex);
  atomic_store_explicit(&state,0,memory_order_release);
  pthread_mutex_unlock(&mutex);
  if(wait)while(atomic_load_explicit(&busy,memory_order_acquire))spin();
  return 0;
}
int pause_spinner(void){return park_spinner(1);}
int pause_spinner_async(void){return park_spinner(0);}
int spinner_busy(void){return atomic_load_explicit(&busy,memory_order_acquire);}
int spinner_active(void){return atomic_load_explicit(&state,memory_order_acquire)==1;}
int spinner_tid(void){return atomic_load(&tid);}
int stop_spinner(void){
  if(!started)return 0;
  pause_spinner();
  pthread_mutex_lock(&mutex);atomic_store_explicit(&state,2,memory_order_release);
  pthread_cond_signal(&condition);pthread_mutex_unlock(&mutex);
  int ret=pthread_join(thread,0);if(!ret)started=0;return ret;
}
