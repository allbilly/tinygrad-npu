/* FP32 MLA dots: independent output lanes, sequential reduction order. */
#include <stddef.h>
#ifndef MLA_CAUSAL_WORK
#define MLA_CAUSAL_WORK 1
#endif
typedef float M4 __attribute__((vector_size(16),aligned(1)));
typedef __fp16 MLAH4 __attribute__((vector_size(8),aligned(1)));
typedef int (*mla_parallel_fn)(void (*)(void**,int,int),void**,int);
typedef struct {
  float *out;
  const float *input,*input_rope,*cache,*packed;
  int heads,capacity,rows,start,context,width,nope,rope,values,stride;
  float scale;
  int done[4];
  int causal_safe;
} MLAWork;

/* Restrict the shortcut to a range that cannot overflow a width<=512 dot
   and scale<=1024. Other IEEE inputs retain the original full calculation. */
static int mla_causal_safe_values(const float *x,size_t count) {
  for(size_t i=0;i<count;i++) {
    union {float f;unsigned u;} bits={.f=x[i]};
    if((bits.u&0x7fffffffU)>0x58800000U)return 0; /* abs(x)>2^50, Inf or NaN */
  }
  return 1;
}
static int mla_tail_needed(M4 x) {
  union {M4 f;unsigned u[4];} bits={.f=x};
  for(int i=0;i<4;i++)if(bits.u[i]==0x80000000U||(bits.u[i]&0x7f800000U)==0x7f800000U)return 1;
  return 0;
}
static void mla_qk_part(MLAWork *w,int group,int groups) {
  const int total=w->start+w->rows;
  for(int head=group;head<w->heads;head+=groups)for(int row=0;row<w->rows;row++) {
    const float *q=w->input+((size_t)head*w->capacity+row)*w->nope;
    const float *qr=w->rope?w->input_rope+((size_t)head*w->capacity+row)*w->rope:q;
    float *out=w->out+((size_t)head*w->capacity+row)*w->stride;
    int end=total;
    if(w->causal_safe&&mla_causal_safe_values(q,w->nope)&&mla_causal_safe_values(qr,w->rope))end=w->start+row+1;
    /* Four independent vectors hide FMA latency; padding avoids power-of-two strides. */
    for(int key=0;key<end;key+=16) {
      M4 sum0={0,0,0,0};
      M4 sum1={0,0,0,0};
      M4 sum2={0,0,0,0};
      M4 sum3={0,0,0,0};
      for(int c=0;c<w->nope;c++) {
        M4 query={q[c],q[c],q[c],q[c]};
        M4 k0=*(const M4*)(w->packed+(size_t)c*(((w->context+15)&~15)+4)+key+0);
        sum0=sum0+query*k0;
        M4 k1=*(const M4*)(w->packed+(size_t)c*(((w->context+15)&~15)+4)+key+4);
        sum1=sum1+query*k1;
        M4 k2=*(const M4*)(w->packed+(size_t)c*(((w->context+15)&~15)+4)+key+8);
        sum2=sum2+query*k2;
        M4 k3=*(const M4*)(w->packed+(size_t)c*(((w->context+15)&~15)+4)+key+12);
        sum3=sum3+query*k3;
      }
      /* The rope row continues the same sequential c reduction, so feeding it
         from a second pointer keeps the contiguous-query accumulation bits. */
      for(int c=0;c<w->rope;c++) {
        M4 query={qr[c],qr[c],qr[c],qr[c]};
        M4 k0=*(const M4*)(w->packed+(size_t)(c+w->nope)*(((w->context+15)&~15)+4)+key+0);
        sum0=sum0+query*k0;
        M4 k1=*(const M4*)(w->packed+(size_t)(c+w->nope)*(((w->context+15)&~15)+4)+key+4);
        sum1=sum1+query*k1;
        M4 k2=*(const M4*)(w->packed+(size_t)(c+w->nope)*(((w->context+15)&~15)+4)+key+8);
        sum2=sum2+query*k2;
        M4 k3=*(const M4*)(w->packed+(size_t)(c+w->nope)*(((w->context+15)&~15)+4)+key+12);
        sum3=sum3+query*k3;
      }
      for(int lane=0;lane<4&&key+0+lane<total;lane++)
        out[key+0+lane]=w->capacity==1?sum0[lane]*w->scale:
          sum0[lane]*w->scale+(key+0+lane>w->start+row?-__builtin_inff():0.0f);
      for(int lane=0;lane<4&&key+4+lane<total;lane++)
        out[key+4+lane]=w->capacity==1?sum1[lane]*w->scale:
          sum1[lane]*w->scale+(key+4+lane>w->start+row?-__builtin_inff():0.0f);
      for(int lane=0;lane<4&&key+8+lane<total;lane++)
        out[key+8+lane]=w->capacity==1?sum2[lane]*w->scale:
          sum2[lane]*w->scale+(key+8+lane>w->start+row?-__builtin_inff():0.0f);
      for(int lane=0;lane<4&&key+12+lane<total;lane++)
        out[key+12+lane]=w->capacity==1?sum3[lane]*w->scale:
          sum3[lane]*w->scale+(key+12+lane>w->start+row?-__builtin_inff():0.0f);
    }
    for(int key=(end+15)&~15;key<total;key++)out[key]=-__builtin_inff();
  }
}
static void mla_pv_part(MLAWork *w,int group,int groups) {
  const int total=w->start+w->rows;
  for(int head=group;head<w->heads;head+=groups)for(int row=0;row<w->rows;row+=4) {
    int count=w->rows-row<4?w->rows-row:4;
    const float *p0=w->input+((size_t)head*w->capacity+row)*w->stride;
    int end=total;
    if(w->causal_safe&&w->start+row+count<total) {
      int candidate=w->start+row+count,zero_tail=1;
      for(int r=0;r<count&&zero_tail;r++)for(int key=candidate;key<total;key++)
        if(p0[r*w->stride+key]!=0.0f){zero_tail=0;break;}
      if(zero_tail)end=candidate;
    }
#if MLA_FP16
    if(count==1) {
      for(int c=0;c<w->values;c+=4) {
        MLAH4 a={0,0,0,0};
        for(int key=0;key<end;key++) {
          MLAH4 value=__builtin_convertvector(*(const M4*)(w->cache+(size_t)key*w->width+c),MLAH4);
          __fp16 p=(__fp16)p0[key];
          a=a+value*(MLAH4){p,p,p,p};
        }
        *(M4*)(w->out+((size_t)head*w->capacity+row)*w->values+c)=__builtin_convertvector(a,M4);
      }
      continue;
    }
#endif
    for(int c=0;c<w->values;c+=4) {
      M4 a0={0,0,0,0},a1={0,0,0,0},a2={0,0,0,0},a3={0,0,0,0};
      for(int key=0;key<end;key++) {
        M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
        float p=p0[key];a0=a0+(M4){p,p,p,p}*value;
        if(count>1){p=p0[w->stride+key];a1=a1+(M4){p,p,p,p}*value;}
        if(count>2){p=p0[2*w->stride+key];a2=a2+(M4){p,p,p,p}*value;}
        if(count>3){p=p0[3*w->stride+key];a3=a3+(M4){p,p,p,p}*value;}
      }
      /* Preserve signed-zero and non-finite accumulator behavior exactly.
         Finite nonzero accumulators cannot change when adding zero products. */
      if(end<total&&(mla_tail_needed(a0)||(count>1&&mla_tail_needed(a1))||
          (count>2&&mla_tail_needed(a2))||(count>3&&mla_tail_needed(a3)))) {
        for(int key=end;key<total;key++) {
          M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
          float p=p0[key];a0=a0+(M4){p,p,p,p}*value;
          if(count>1){p=p0[w->stride+key];a1=a1+(M4){p,p,p,p}*value;}
          if(count>2){p=p0[2*w->stride+key];a2=a2+(M4){p,p,p,p}*value;}
          if(count>3){p=p0[3*w->stride+key];a3=a3+(M4){p,p,p,p}*value;}
        }
      }
      float *out=w->out+((size_t)head*w->capacity+row)*w->values+c;
      *(M4*)out=a0;
      if(count>1)*(M4*)(out+w->values)=a1;
      if(count>2)*(M4*)(out+2*w->values)=a2;
      if(count>3)*(M4*)(out+3*w->values)=a3;
    }
  }
}

static void mla_qk_worker(void **args,int unused,int group) {
  (void)unused;MLAWork *w=args[0];if(group<0||group>=4)return;
  mla_qk_part(w,group,4);w->done[group]=1;
}
static void mla_pv_worker(void **args,int unused,int group) {
  (void)unused;MLAWork *w=args[0];if(group<0||group>=4)return;
  mla_pv_part(w,group,4);w->done[group]=1;
}
static void mla_execute(MLAWork *w,mla_parallel_fn parallel,int qk) {
  if(w->start<0||w->rows<1||w->rows>w->capacity||w->start+w->rows>w->context)__builtin_trap();
  /* Single-row decode keeps its existing work and avoids the safety scan. */
  w->causal_safe=MLA_CAUSAL_WORK&&w->capacity>1&&w->scale>0.0f&&w->scale<=1024.0f&&
    mla_causal_safe_values(w->cache,(size_t)(w->start+w->rows)*w->width);
  if(!parallel){if(qk)mla_qk_part(w,0,1);else mla_pv_part(w,0,1);return;}
  void *args[4]={w,0,0,0};
  if(parallel(qk?mla_qk_worker:mla_pv_worker,args,0))__builtin_trap();
  for(int i=0;i<4;i++)if(!w->done[i])__builtin_trap();
}
