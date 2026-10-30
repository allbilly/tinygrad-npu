#include <stddef.h>
typedef float F4 __attribute__((ext_vector_type(4),aligned(1)));
typedef __fp16 H4 __attribute__((ext_vector_type(4),aligned(1)));
extern int pthread_create(unsigned long *,const void *,void *(*)(void *),void *);
extern int pthread_join(unsigned long,void **);

static void put(float *out,F4 v,int count) {
  if(count>=4)*(F4*)out=v;
  else for(int i=0;i<count;i++)out[i]=v[i];
}
static void part(float *out,const float *x,const __fp16 *w,int rows,int k,int n,int padded,int first,int last) {
  for(int column=first;column<last;column+=8) {
    int left=n-column<8?n-column:8;
    int row=0;
    for(;row+4<=rows;row+=4) {
      F4 a0={0},a1={0},b0={0},b1={0},c0={0},c1={0},d0={0},d1={0};
      const float *xa=x+(size_t)row*k,*xb=xa+k,*xc=xb+k,*xd=xc+k;
      for(int depth=0;depth<k;depth++) {
        const __fp16 *p=w+(size_t)(column/8)*k*8+depth*8;
        F4 lo=__builtin_convertvector(*(const H4*)p,F4),hi=__builtin_convertvector(*(const H4*)(p+4),F4);
        a0=a0+xa[depth]*lo;a1=a1+xa[depth]*hi;
        b0=b0+xb[depth]*lo;b1=b1+xb[depth]*hi;
        c0=c0+xc[depth]*lo;c1=c1+xc[depth]*hi;
        d0=d0+xd[depth]*lo;d1=d1+xd[depth]*hi;
      }
      float *p=out+(size_t)row*n+column;
      put(p,a0,left<4?left:4);if(left>4)put(p+4,a1,left-4);
      put(p+n,b0,left<4?left:4);if(left>4)put(p+n+4,b1,left-4);
      put(p+2*n,c0,left<4?left:4);if(left>4)put(p+2*n+4,c1,left-4);
      put(p+3*n,d0,left<4?left:4);if(left>4)put(p+3*n+4,d1,left-4);
    }
    for(;row<rows;row++) {
      F4 lo={0},hi={0};const float *a=x+(size_t)row*k;
      for(int depth=0;depth<k;depth++) {
        const __fp16 *p=w+(size_t)(column/8)*k*8+depth*8;
        lo=lo+a[depth]*__builtin_convertvector(*(const H4*)p,F4);
        hi=hi+a[depth]*__builtin_convertvector(*(const H4*)(p+4),F4);
      }
      float *p=out+(size_t)row*n+column;put(p,lo,left<4?left:4);if(left>4)put(p+4,hi,left-4);
    }
  }
}
typedef struct {float *out;const float *x;const __fp16 *w;int rows,k,n,padded,first,last;} Work;
static void *worker(void *p) {
  Work *w=p;part(w->out,w->x,w->w,w->rows,w->k,w->n,w->padded,w->first,w->last);return (void*)0;
}
static void static_gemm(float *out,const float *x,const __fp16 *w,int rows,int k,int n,int threads) {
  const int padded=(n+7)/8*8,blocks=padded/8;
  if(threads!=4 || n<128) {part(out,x,w,rows,k,n,padded,0,padded);return;}
  Work work[4];unsigned long ids[3];int created[3];
  for(int i=0;i<4;i++)work[i]=(Work){out,x,w,rows,k,n,padded,blocks*i/4*8,blocks*(i+1)/4*8};
  for(int i=0;i<3;i++){created[i]=pthread_create(ids+i,(void*)0,worker,work+i)==0;if(!created[i])worker(work+i);}
  worker(work+3);for(int i=0;i<3;i++)if(created[i])pthread_join(ids[i],(void**)0);
}
