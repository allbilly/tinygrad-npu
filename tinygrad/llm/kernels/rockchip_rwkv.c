/* RWKV7: old-state rank-one correction precedes decay, unlike GatedDeltaNet. */
#include <stddef.h>
#ifndef RWKV_ORDERED_NORMALIZATION
#define RWKV_ORDERED_NORMALIZATION 0
#endif
#ifndef RWKV_SCAN_WIDTH
#define RWKV_SCAN_WIDTH 0
#endif
typedef float F4 __attribute__((vector_size(16), aligned(1)));

static F4 load_values(const float *p,int count) {
  if(count==4)return *(const F4*)p;
  return (F4){p[0],count>1?p[1]:0,count>2?p[2]:0,0};
}
static void store_values(float *p,F4 value,int count) {
  if(count==4){*(F4*)p=value;return;}
  for(int i=0;i<count;i++)p[i]=value[i];
}

typedef struct {
  float *out,*state;
  const float *r,*k,*v,*decay,*rate,*unit,*unit_weight,*unit_norm;
  int start,rows,batch,heads,capacity,width;
  int done[4];
} RWKVWork;

static void rwkv_scan_part(RWKVWork *w,int group,int groups) {
  const int width=RWKV_SCAN_WIDTH?RWKV_SCAN_WIDTH:w->width;
  float columns[256*256] __attribute__((aligned(16)));
  for(int bh=group;bh<w->batch*w->heads;bh+=groups) {
    const int batch=bh/w->heads,head=bh%w->heads;
    float *state=w->state+(size_t)bh*width*width;
    for(int c=0;c<width;c++)for(int j=0;j<width;j++)
      columns[c*width+j]=w->start?state[j*width+c]:0;
    for(int t=0;t<w->rows;t++) {
      const size_t step=((size_t)batch*w->capacity*w->heads+t*w->heads+head)*width;
      const float *r=w->r+step,*k=w->k+step,*v=w->v+step;
      const float *decay=w->decay+step,*rate=w->rate+step,*unit=w->unit+step;
#if RWKV_SCAN_WIDTH == 64
      for(int j=0;j<width;j+=16) {
        F4 predicted[4]={(F4){0},(F4){0},(F4){0},(F4){0}};
        const int parts=(width-j+3)/4<4?(width-j+3)/4:4;
        for(int c=0;c<width;c++) {
          const float factor=RWKV_ORDERED_NORMALIZATION?w->unit_weight[head*width+c]:1;
          #pragma clang loop unroll(full)
          for(int q=0;q<4;q++)if(q<parts) {
            const int offset=j+4*q,count=width-offset<4?width-offset:4;
            F4 old=load_values(columns+c*width+offset,count);
            if(RWKV_ORDERED_NORMALIZATION)
              predicted[q]=predicted[q]+(old*(F4){unit[c],unit[c],unit[c],unit[c]})*(F4){factor,factor,factor,factor};
            else predicted[q]=predicted[q]+old*(F4){unit[c],unit[c],unit[c],unit[c]};
          }
        }
        if(RWKV_ORDERED_NORMALIZATION) {
          float norm=w->unit_norm[((size_t)batch*w->capacity+t)*w->heads+head];
          #pragma clang loop unroll(full)
          for(int q=0;q<4;q++)predicted[q]=predicted[q]/(F4){norm,norm,norm,norm};
        }
        F4 value[4],result[4]={(F4){0},(F4){0},(F4){0},(F4){0}};
        #pragma clang loop unroll(full)
        for(int q=0;q<4;q++)if(q<parts) {
          const int offset=j+4*q,count=width-offset<4?width-offset:4;
          value[q]=load_values(v+offset,count);
        }
        for(int c=0;c<width;c++) {
          const float correction=RWKV_ORDERED_NORMALIZATION?rate[c]:unit[c]*rate[c];
          #pragma clang loop unroll(full)
          for(int q=0;q<4;q++)if(q<parts) {
            const int offset=j+4*q,count=width-offset<4?width-offset:4;
            float *p=columns+c*width+offset;
            F4 old=load_values(p,count);
            F4 updated=(old*(F4){decay[c],decay[c],decay[c],decay[c]}-
              predicted[q]*(F4){correction,correction,correction,correction})+value[q]*(F4){k[c],k[c],k[c],k[c]};
            store_values(p,updated,count);
            result[q]=result[q]+updated*(F4){r[c],r[c],r[c],r[c]};
          }
        }
        #pragma clang loop unroll(full)
        for(int q=0;q<4;q++)if(q<parts) {
          const int offset=j+4*q,count=width-offset<4?width-offset:4;
          store_values(w->out+step+offset,result[q],count);
        }
      }
#else
      for(int j=0;j<width;j+=4) {
        const int count=width-j<4?width-j:4;
        F4 predicted=(F4){0,0,0,0};
        for(int c=0;c<width;c++) {
          F4 old=load_values(columns+c*width+j,count);
          if(RWKV_ORDERED_NORMALIZATION) {
            float factor=w->unit_weight[head*width+c];
            predicted=predicted+(old*(F4){unit[c],unit[c],unit[c],unit[c]})*(F4){factor,factor,factor,factor};
          } else predicted=predicted+old*(F4){unit[c],unit[c],unit[c],unit[c]};
        }
        if(RWKV_ORDERED_NORMALIZATION) {
          float norm=w->unit_norm[((size_t)batch*w->capacity+t)*w->heads+head];
          predicted=predicted/(F4){norm,norm,norm,norm};
        }
        const F4 value=load_values(v+j,count);
        F4 result=(F4){0,0,0,0};
        for(int c=0;c<width;c++) {
          float *p=columns+c*width+j;
          const float correction=RWKV_ORDERED_NORMALIZATION?rate[c]:unit[c]*rate[c];
          F4 old=load_values(p,count);
          F4 updated=(old*(F4){decay[c],decay[c],decay[c],decay[c]}-
            predicted*(F4){correction,correction,correction,correction})+value*(F4){k[c],k[c],k[c],k[c]};
          store_values(p,updated,count);
          result=result+updated*(F4){r[c],r[c],r[c],r[c]};
        }
        store_values(w->out+step+j,result,count);
      }
#endif
    }
    for(int j=0;j<width;j++)for(int c=0;c<width;c++)state[j*width+c]=columns[c*width+j];
    for(int t=w->rows;t<w->capacity;t++) {
      const size_t step=((size_t)batch*w->capacity*w->heads+t*w->heads+head)*width;
      for(int j=0;j<width;j++)w->out[step+j]=0;
    }
  }
}

static void rwkv_pool_part(void **args,int unused,int group) {
  (void)unused;
  RWKVWork *w=args[0];
  if(group<0||group>=4)return;
  rwkv_scan_part(w,group,4);w->done[group]=1;
}
typedef int (*rwkv_parallel_fn)(void (*)(void**,int,int),void**,int);

static void rwkv_scan(float *out,const float *r,const float *k,const float *v,
    const float *decay,const float *rate,const float *unit,float *state,int start,int rows,
    int batch,int heads,int capacity,int width,rwkv_parallel_fn parallel,const float *unit_weight,const float *unit_norm) {
  if(start<0||rows<1||rows>capacity||capacity>256||width<1||width>256||batch<1||heads<1)__builtin_trap();
  RWKVWork work={out,state,r,k,v,decay,rate,unit,unit_weight,unit_norm,start,rows,batch,heads,capacity,width,{0}};
  if(parallel) {
    void *args[4]={&work,0,0,0};
    /* A dispatch error may follow partial state writes: never retry the scan. */
    if(parallel(rwkv_pool_part,args,0))__builtin_trap();
    for(int i=0;i<4;i++)if(!work.done[i])__builtin_trap();
  } else rwkv_scan_part(&work,0,1);
}
