#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
extern int cpu_gated_delta_scan_part(float*,const float*,const float*,const float*,const float*,
  const float*,float*,int,int,int,int,int,int,int,int,float*,int,int);
static const int B=1,H=32,T=128,K=128,V=128,A=1;
static float *q,*k,*v,*be,*al,*st,*o;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static void *run(void *p){ long g=(long)p;
  cpu_set_t s; CPU_ZERO(&s); CPU_SET(4+g,&s); pthread_setaffinity_np(pthread_self(),sizeof(s),&s);
  cpu_gated_delta_scan_part(o,q,k,v,be,al,st,0,T,B,H,T,K,V,A,NULL,(int)g,4); return 0; }
int main(void){
  size_t nq=(size_t)B*H*T*K,na=(size_t)B*H*T*A,ns=(size_t)B*H*V*K,no=(size_t)B*H*T*V;
  q=malloc(nq*4);k=malloc(nq*4);v=malloc(nq*4);be=malloc(na*4);al=malloc(na*4);st=malloc(ns*4);o=malloc(no*4);
  for(size_t i=0;i<nq;i++){q[i]=k[i]=v[i]=0.01f*(i%7);} for(size_t i=0;i<na;i++){be[i]=0.5f;al[i]=0.9f;}
  memset(st,0,ns*4); int reps=5;
  for(int n=1;n<=4;n*=2){
    pthread_t t[4]; double a=now();
    for(int r=0;r<reps;r++){ for(int g=0;g<n;g++) pthread_create(&t[g],0,run,(void*)(long)g);
      for(int g=0;g<n;g++) pthread_join(t[g],0); }
    double b=now();
    printf("%d thread(s), %d group(s): %6.2f ms  (per group %.2f)\n",n,n,(b-a)/reps,(b-a)/reps/n);
  }
  return 0;
}
