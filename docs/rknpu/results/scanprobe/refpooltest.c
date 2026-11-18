#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef int (*scan_parallel_fn)(void (*)(void **, int, int), void **, int);
extern int cpu_gated_delta_scan_pooled(float*,const float*,const float*,const float*,const float*,
  const float*,float*,int,int,int,int,int,int,int,int,float*,scan_parallel_fn,int*);
extern int cpu_gated_delta_scan(float*,const float*,const float*,const float*,const float*,
  const float*,float*,int,int,int,int,int,int,int,int,float*);
extern int start_pool(int cpu);
extern int parallel(scan_parallel_fn, void **, int);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int main(void){
  const int B=1,H=32,T=128,K=128,V=128,A=1;
  size_t nq=(size_t)B*H*T*K,na=(size_t)B*H*T*A,ns=(size_t)B*H*V*K,no=(size_t)B*H*T*V;
  float *q=malloc(nq*4),*k=malloc(nq*4),*v=malloc(nq*4),*be=malloc(na*4),*al=malloc(na*4);
  float *s0=malloc(ns*4),*s1=malloc(ns*4),*o0=malloc(no*4),*o1=malloc(no*4);
  for(size_t i=0;i<nq;i++){q[i]=k[i]=v[i]=0.01f*(i%7);} for(size_t i=0;i<na;i++){be[i]=0.5f;al[i]=0.9f;}
  memset(s0,0,ns*4); memset(s1,0,ns*4); memset(o0,0,no*4); memset(o1,0,no*4);
  int st=start_pool(4); printf("start_pool(4) = %d\n", st);
  int status=-1, reps=5;
  int rc=cpu_gated_delta_scan_pooled(o1,q,k,v,be,al,s1,0,T,B,H,T,K,V,A,NULL,(scan_parallel_fn)parallel,&status);
  cpu_gated_delta_scan(o0,q,k,v,be,al,s0,0,T,B,H,T,K,V,A,NULL);
  int diff=0; for(size_t i=0;i<no;i++) if(memcmp(&o0[i],&o1[i],4)) diff++;
  for(size_t i=0;i<ns;i++) if(memcmp(&s0[i],&s1[i],4)) diff++;
  printf("rc=%d status=%d differing=%d\n", rc, status, diff);
  double a=now(); for(int r=0;r<reps;r++) cpu_gated_delta_scan(o0,q,k,v,be,al,s0,0,T,B,H,T,K,V,A,NULL); double b=now();
  for(int r=0;r<reps;r++) cpu_gated_delta_scan_pooled(o1,q,k,v,be,al,s1,0,T,B,H,T,K,V,A,NULL,(scan_parallel_fn)parallel,&status); double c=now();
  printf("serial %.2f ms/call   refpool %.2f ms/call   speedup %.2fx   status=%d\n",(b-a)/reps,(c-b)/reps,(b-a)/(c-b),status);
  return 0;
}
