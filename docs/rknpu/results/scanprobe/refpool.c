#define _GNU_SOURCE
#include <stdatomic.h>
#include <stdint.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>

typedef void (*part_fn)(void**,int,int);
typedef struct {pthread_t thread;_Atomic int ready,tid;int error,cpu,group;char padding[32];} Worker;
static Worker workers[3];
static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition=PTHREAD_COND_INITIALIZER;
static _Atomic int mode,remaining,busy,exited,parked;
static _Atomic uint64_t generation;
static int started,position,event_wait;
static part_fn function;
static void *arguments[4];
static uint64_t calls;
static inline void spin(void){__asm__ volatile("yield" ::: "memory");}
static inline void wake_idle(void){if(event_wait)__asm__ volatile("sev" ::: "memory");}
static inline void idle_wait(void){if(event_wait)__asm__ volatile("wfe" ::: "memory");else spin();}
static void await_exit(int count){if(!event_wait)return;unsigned tick=0;while(atomic_load_explicit(&exited,memory_order_acquire)<count){if(!(tick++&63))wake_idle();spin();}}
static void await_park(void){if(!event_wait)return;unsigned tick=0;while(atomic_load_explicit(&parked,memory_order_acquire)<3){if(!(tick++&63))wake_idle();spin();}}

static void *run(void *argument){
  Worker *worker=argument;
  uint64_t seen=atomic_load_explicit(&generation,memory_order_acquire);
  cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(worker->cpu,&mask);
  worker->error=pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask);
  atomic_store(&worker->tid,(int)syscall(SYS_gettid));
  atomic_store_explicit(&worker->ready,1,memory_order_release);
  if(worker->error){atomic_fetch_add_explicit(&exited,1,memory_order_release);return 0;}
  for(;;){
    int value=atomic_load_explicit(&mode,memory_order_acquire);
    if(value==2){atomic_fetch_add_explicit(&exited,1,memory_order_release);return 0;}
    if(value==3){
      pthread_mutex_lock(&mutex);
      if(atomic_load_explicit(&mode,memory_order_acquire)==3){
        atomic_fetch_add_explicit(&parked,1,memory_order_release);
        while(atomic_load_explicit(&mode,memory_order_acquire)==3)pthread_cond_wait(&condition,&mutex);
        atomic_fetch_sub_explicit(&parked,1,memory_order_release);
      }
      pthread_mutex_unlock(&mutex);continue;
    }
    uint64_t next=atomic_load_explicit(&generation,memory_order_acquire);
    if(next==seen){idle_wait();continue;}
    seen=next;function(arguments,position,worker->group);
    atomic_fetch_sub_explicit(&remaining,1,memory_order_release);
  }
}
int start_pool(int cpu){
  if(started)return EALREADY;
  if(cpu<0||cpu+2>=CPU_SETSIZE)return EINVAL;
  atomic_store(&mode,3);atomic_store(&exited,0);atomic_store(&parked,0);atomic_store(&remaining,0);atomic_store(&busy,0);atomic_store(&generation,0);
  const char *option=getenv("ONPU_HOST_POOL_WFE");
  event_wait=option&&option[0]=='1'&&!option[1];
  int count=0,error=0;
  for(int i=0;i<3;i++){
    workers[i].cpu=cpu+i;workers[i].group=i+1;atomic_store(&workers[i].ready,0);
    error=pthread_create(&workers[i].thread,0,run,workers+i);if(error)break;count++;
    while(!atomic_load_explicit(&workers[i].ready,memory_order_acquire))spin();
    if((error=workers[i].error))break;
  }
  if(error){
    pthread_mutex_lock(&mutex);atomic_store(&mode,2);pthread_cond_broadcast(&condition);pthread_mutex_unlock(&mutex);
    await_exit(count);
    for(int i=0;i<count;i++)pthread_join(workers[i].thread,0);
    return error;
  }
  started=1;return 0;
}
static int parallel_work(part_fn fn,void **args,int pos,int park){
  int current=atomic_load_explicit(&mode,memory_order_acquire);
  if(!started||current==2||atomic_exchange_explicit(&busy,1,memory_order_acquire))return EBUSY;
  if(current==3)pthread_mutex_lock(&mutex);
  function=fn;position=pos;for(int i=0;i<4;i++)arguments[i]=args[i];
  atomic_store(&remaining,3);atomic_fetch_add_explicit(&generation,1,memory_order_release);
  atomic_store_explicit(&mode,0,memory_order_release);
  if(current==3){pthread_cond_broadcast(&condition);pthread_mutex_unlock(&mutex);}
  wake_idle();fn(args,pos,0);
  unsigned tick=0;while(atomic_load_explicit(&remaining,memory_order_acquire)){if(!(tick++&63))wake_idle();spin();}
  if(park){atomic_store_explicit(&mode,3,memory_order_release);wake_idle();await_park();}
  calls++;atomic_store_explicit(&busy,0,memory_order_release);return 0;
}
int parallel(part_fn fn,void **args,int pos){return parallel_work(fn,args,pos,0);}
int parallel_parked(part_fn fn,void **args,int pos){
  cpu_set_t previous,main_mask;int changed=0;
  if(!pthread_getaffinity_np(pthread_self(),sizeof(previous),&previous)&&CPU_ISSET(4,&previous)&&CPU_COUNT(&previous)>1){
    CPU_ZERO(&main_mask);CPU_SET(4,&main_mask);
    changed=!pthread_setaffinity_np(pthread_self(),sizeof(main_mask),&main_mask);
  }
  int result=parallel_work(fn,args,pos,1);
  if(changed){int error=pthread_setaffinity_np(pthread_self(),sizeof(previous),&previous);if(!result)result=error;}
  return result;
}
int pause_pool(void){
  if(!started)return 0;
  if(atomic_load_explicit(&busy,memory_order_acquire))return EBUSY;
  atomic_store_explicit(&mode,3,memory_order_release);wake_idle();await_park();return 0;
}
uint64_t pool_calls(void){return calls;}
int pool_parked(void){return atomic_load_explicit(&mode,memory_order_acquire)==3;}
int pool_tid(void){return atomic_load(&workers[0].tid);}
int pool_thread_tid(int index){return index>=0&&index<3?atomic_load(&workers[index].tid):-1;}
int stop_pool(void){
  if(!started)return 0;
  if(atomic_load_explicit(&busy,memory_order_acquire))return EBUSY;
  pthread_mutex_lock(&mutex);atomic_store_explicit(&mode,2,memory_order_release);
  pthread_cond_broadcast(&condition);pthread_mutex_unlock(&mutex);
  await_exit(3);
  int error=0;for(int i=0;i<3;i++){int result=pthread_join(workers[i].thread,0);if(result)error=result;}
  if(!error)started=0;
  return error;
}

int pool_wait_mode(void){return event_wait;}
int pool_sleepers(void){return atomic_load_explicit(&parked,memory_order_acquire);}

// Change the policy only while every worker is parked and no job can publish.
int set_pool_wait_mode(int enabled){
 if(enabled!=0&&enabled!=1)return EINVAL;
 if(!started)return ENODEV;
 if(atomic_exchange_explicit(&busy,1,memory_order_acquire))return EBUSY;
 if(atomic_load_explicit(&mode,memory_order_acquire)!=3){atomic_store_explicit(&busy,0,memory_order_release);return EBUSY;}
 unsigned tick=0;
 while(atomic_load_explicit(&parked,memory_order_acquire)<3){if(!(tick++&63))wake_idle();spin();}
 pthread_mutex_lock(&mutex);
 event_wait=enabled;
 pthread_mutex_unlock(&mutex);
 atomic_store_explicit(&busy,0,memory_order_release);return 0;
}
