#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef int (*scan_parallel_fn)(void (*)(void **, int, int), void **, int);
extern int cpu_gated_delta_scan_pooled(float *, const float *, const float *, const float *,
  const float *, const float *, float *, int, int, int, int, int, int, int, int, float *,
  scan_parallel_fn, int *);
extern int rk_run(void (*)(void **, int, int), void **, int);
extern double rk_group_start(int), rk_group_end(int);
extern int rk_group_cpu(int);
int main(void) {
  const int B=1,H=32,T=128,K=128,V=128,A=1;
  size_t nq=(size_t)B*H*T*K, na=(size_t)B*H*T*A, ns=(size_t)B*H*V*K, no=(size_t)B*H*T*V;
  float *q=calloc(nq,4),*k=calloc(nq,4),*v=calloc(nq,4),*be=calloc(na,4),*al=calloc(na,4);
  float *st=calloc(ns,4),*o=calloc(no,4); int status=0;
  for (int r=0;r<3;r++)
    cpu_gated_delta_scan_pooled(o,q,k,v,be,al,st,0,T,B,H,T,K,V,A,NULL,(scan_parallel_fn)rk_run,&status);
  double s[4],e[4]; int n=status==1?4:1;
  for (int i=0;i<n;i++){s[i]=rk_group_start(i);e[i]=rk_group_end(i);}
  double lo=s[0],hi=e[0]; for(int i=1;i<n;i++){if(s[i]<lo)lo=s[i]; if(e[i]>hi)hi=e[i];}
  double sum=0; for(int i=0;i<n;i++) sum+=e[i]-s[i];
  printf("status=%d groups=%d wall=%.2f ms cpu=%.2f ms overlap=%.2fx\n",status,n,hi-lo,sum,sum/(hi-lo));
  for(int i=0;i<n;i++) printf("  group %d on cpu%d: %7.2f -> %7.2f  (%.2f ms)\n",i,rk_group_cpu(i),s[i],e[i],e[i]-s[i]);
  return 0;
}
