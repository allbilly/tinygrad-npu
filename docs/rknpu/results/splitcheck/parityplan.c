#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int (*scan_parallel_fn_fn)(void (*)(void **, int, int), void **, int);
extern int cpu_gated_delta_scan(float*,const float*,const float*,const float*,const float*,
  const float*,float*,int,int,int,int,int,int,int,int,float*);
extern int cpu_gated_delta_scan_part(float*,const float*,const float*,const float*,const float*,
  const float*,float*,int,int,int,int,int,int,int,int,float*,int,int);
extern int cpu_gated_delta_scan_pooled(float*,const float*,const float*,const float*,const float*,
  const float*,float*,int,int,int,int,int,int,int,int,float*,scan_parallel_fn_fn,int*);
extern int rk_run(void (*)(void**,int,int), void**, int);

static const int B=1,H=32,T=128,K=128,V=128,A=1;
static float *q,*k,*v,*be,*al;
static void fill(void){ size_t nq=(size_t)B*H*T*K,na=(size_t)B*H*T*A; srand(5);
  q=malloc(nq*4);k=malloc(nq*4);v=malloc(nq*4);be=malloc(na*4);al=malloc(na*4);
  for(size_t i=0;i<nq;i++){q[i]=((float)rand()/RAND_MAX-0.5f)*0.2f;k[i]=((float)rand()/RAND_MAX-0.5f)*0.2f;v[i]=((float)rand()/RAND_MAX-0.5f)*0.2f;}
  for(size_t i=0;i<na;i++){be[i]=0.5f;al[i]=0.9f;} }
static void st0(float *s){ size_t n=(size_t)B*H*V*K; for(size_t i=0;i<n;i++) s[i]=0.01f*(i%13); }
int main(void){
  fill();
  size_t no=(size_t)B*H*T*V, ns=(size_t)B*H*V*K;
  float *o1=calloc(no,4),*o2=calloc(no,4),*o3=calloc(no,4);
  float *s1=malloc(ns*4),*s2=malloc(ns*4),*s3=malloc(ns*4);
  st0(s1); st0(s2); st0(s3);
  cpu_gated_delta_scan(o1,q,k,v,be,al,s1,0,T,B,H,T,K,V,A,NULL);
  for(int g=0;g<4;g++) cpu_gated_delta_scan_part(o2,q,k,v,be,al,s2,0,T,B,H,T,K,V,A,NULL,g,4);
  int st=-1;
  int rc=cpu_gated_delta_scan_pooled(o3,q,k,v,be,al,s3,0,T,B,H,T,K,V,A,NULL,(scan_parallel_fn_fn)rk_run,&st);
  struct { const char *name; float *o; float *s; } sets[] = {{"parts-sequential",o2,s2},{"pooled",o3,s3}};
  for (int t=0;t<2;t++){
    size_t fd=(size_t)-1; int nd=0, heads[32]={0};
    for(size_t i=0;i<no;i++) if(memcmp(&o1[i],&sets[t].o[i],4)){ if(fd==(size_t)-1)fd=i; nd++; heads[i/((size_t)T*V)]=1; }
    printf("%-16s out differing=%7d first_elem=%zu first_head=%zu  heads_touched:",sets[t].name,nd,fd,fd/((size_t)T*V));
    for(int h=0;h<H;h++) if(heads[h]) printf(" %d",h);
    size_t sfd=(size_t)-1; int snd=0;
    for(size_t i=0;i<ns;i++) if(memcmp(&s1[i],&sets[t].s[i],4)){ if(sfd==(size_t)-1)sfd=i; snd++; }
    printf("  | state differing=%d first=%zu\n", snd, sfd);
  }
  printf("pooled rc=%d status=%d\n", rc, st);
  return 0;
}
