#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef void rk_fn_t(void **, int, int);
extern int rk_run(rk_fn_t *, void **, int);

void original_kernel(unsigned char *);
void transformed_kernel(unsigned char *);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int main(int argc, char **argv){
  int reps = argc > 1 ? atoi(argv[1]) : 10;
  size_t n_out = 524288, n_1 = 1073152, n_2 = 131072, n_3 = 2048;
  float *o0 = calloc(n_out,4), *o1 = calloc(n_out,4);
  float *a1 = malloc(n_1*4), *b1 = malloc(n_1*4);
  unsigned char *a2 = malloc(n_2), *b2 = malloc(n_2);
  float *a3 = malloc(n_3*4), *b3 = malloc(n_3*4);
  srand(11);
  for (size_t i=0;i<n_1;i++){ float v=((float)rand()/RAND_MAX-0.5f)*8.0f; a1[i]=b1[i]=v; }
  for (size_t i=0;i<n_2;i++){ unsigned char v=(unsigned char)(rand()&0xff); a2[i]=b2[i]=v; }
  for (size_t i=0;i<n_3;i++){ float v=((float)rand()/RAND_MAX); a3[i]=b3[i]=v; }

  unsigned long p0[4]={(unsigned long)o0,(unsigned long)a1,(unsigned long)a2,(unsigned long)a3};
  unsigned long p1[4]={(unsigned long)o1,(unsigned long)b1,(unsigned long)b2,(unsigned long)b3};
  original_kernel((unsigned char*)p0);
  transformed_kernel((unsigned char*)p1);
  int diff=0; for(size_t i=0;i<n_out;i++) if(memcmp(&o0[i],&o1[i],4)) diff++;
  printf("differing=%d verdict=%s\n", diff, diff?"FAIL":"BITWISE-EQUAL");
  double ts=0,tp=0;
  for(int r=0;r<reps;r++){
    double a=now(); original_kernel((unsigned char*)p0); double b=now();
    transformed_kernel((unsigned char*)p1); double c=now();
    ts+=b-a; tp+=c-b;
  }
  printf("reps=%d  serial %.2f ms/call  pooled %.2f ms/call  speedup %.2fx\n", reps, ts/reps, tp/reps, ts/tp);
  return diff?1:0;
}
