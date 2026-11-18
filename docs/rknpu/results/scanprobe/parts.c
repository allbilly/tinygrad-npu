#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern int cpu_gated_delta_scan_part(float*,const float*,const float*,const float*,const float*,
  const float*,float*,int,int,int,int,int,int,int,int,float*,int,int);
extern int cpu_gated_delta_scan(float*,const float*,const float*,const float*,const float*,
  const float*,float*,int,int,int,int,int,int,int,int,float*);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int main(void){
  const int B=1,H=32,T=128,K=128,V=128,A=1;
  size_t nq=(size_t)B*H*T*K, na=(size_t)B*H*T*A, ns=(size_t)B*H*V*K, no=(size_t)B*H*T*V;
  float *q=malloc(nq*4),*k=malloc(nq*4),*v=malloc(nq*4),*be=malloc(na*4),*al=malloc(na*4);
  float *st=malloc(ns*4),*st2=malloc(ns*4),*o=malloc(no*4);
  for(size_t i=0;i<nq;i++){q[i]=k[i]=v[i]=0.01f*(i%7);} for(size_t i=0;i<na;i++){be[i]=0.5f;al[i]=0.9f;}
  memset(st,0,ns*4); memset(st2,0,ns*4);
  int reps=5; double a,b,c,d,e;
  a=now(); for(int r=0;r<reps;r++) cpu_gated_delta_scan(o,q,k,v,be,al,st,0,T,B,H,T,K,V,A,NULL); b=now();
  printf("all 32 heads, parts=1 : %6.2f ms/call\n",(b-a)/reps);
  a=now(); for(int r=0;r<reps;r++) cpu_gated_delta_scan_part(o,q,k,v,be,al,st2,0,T,B,H,T,K,V,A,NULL,0,4); b=now();
  printf("group 0 of 4 (8 heads): %6.2f ms/call\n",(b-a)/reps);
  a=now(); for(int g=0;g<4;g++){ cpu_gated_delta_scan_part(o,q,k,v,be,al,st2,0,T,B,H,T,K,V,A,NULL,g,4);} 
  for(int r=1;r<reps;r++){ for(int g=0;g<4;g++) cpu_gated_delta_scan_part(o,q,k,v,be,al,st2,0,T,B,H,T,K,V,A,NULL,g,4);} 
  c=now(); printf("4 groups serial in-process: %6.2f ms total\n",(c-b)/reps);
  return 0;
}
