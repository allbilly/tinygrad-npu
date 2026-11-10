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


/* ---- Opt-in head-blocked single-row decode path (MLA_HEAD_BLOCK > 1) ----
   The baseline kernels reload the shared K block (qk) or the latent cache row
   (pv) once PER HEAD. These variants load it once for MLA_HEAD_BLOCK heads;
   every output still accumulates over ascending c (qk) / ascending key (pv)
   with the same multiply/add expression, so results are bit-identical. Only
   the capacity==1, rows==1 shape (decode) uses this path. */
#if MLA_HEAD_BLOCK > 1
#define MLA_HB MLA_HEAD_BLOCK
static void mla_qk_single(MLAWork *w,int head) {
  const int total=w->start+w->rows;const int pstride=((w->context+15)&~15)+4;
  const float *q=w->input+((size_t)head*w->capacity)*w->nope;
  const float *qr=w->rope?w->input_rope+((size_t)head*w->capacity)*w->rope:q;
  float *out=w->out+((size_t)head*w->capacity)*w->stride;
  for(int key=0;key<total;key+=16) {
    M4 s0={0,0,0,0},s1={0,0,0,0},s2={0,0,0,0},s3={0,0,0,0};
    for(int c=0;c<w->nope;c++) {
      M4 query={q[c],q[c],q[c],q[c]};
      const float *base=w->packed+(size_t)c*pstride+key;
      M4 k0=*(const M4*)(base),k1=*(const M4*)(base+4),k2=*(const M4*)(base+8),k3=*(const M4*)(base+12);
      s0=s0+query*k0;s1=s1+query*k1;s2=s2+query*k2;s3=s3+query*k3;
    }
    for(int c=0;c<w->rope;c++) {
      M4 query={qr[c],qr[c],qr[c],qr[c]};
      const float *base=w->packed+(size_t)(c+w->nope)*pstride+key;
      M4 k0=*(const M4*)(base),k1=*(const M4*)(base+4),k2=*(const M4*)(base+8),k3=*(const M4*)(base+12);
      s0=s0+query*k0;s1=s1+query*k1;s2=s2+query*k2;s3=s3+query*k3;
    }
    for(int lane=0;lane<4&&key+lane<total;lane++)out[key+lane]=s0[lane]*w->scale;
    for(int lane=0;lane<4&&key+4+lane<total;lane++)out[key+4+lane]=s1[lane]*w->scale;
    for(int lane=0;lane<4&&key+8+lane<total;lane++)out[key+8+lane]=s2[lane]*w->scale;
    for(int lane=0;lane<4&&key+12+lane<total;lane++)out[key+12+lane]=s3[lane]*w->scale;
  }
}
static void mla_qk_part_hb(MLAWork *w,int group,int groups) {
  const int HB=MLA_HB;const int total=w->start+w->rows;const int pstride=((w->context+15)&~15)+4;
  const int nblocks=(w->heads+HB-1)/HB;
  for(int blk=group;blk<nblocks;blk+=groups) {
    int hb=blk*HB,nh=w->heads-hb;if(nh>HB)nh=HB;
    if(nh<HB){for(int h=0;h<nh;h++)mla_qk_single(w,hb+h);continue;}
    const size_t hstep=(size_t)w->capacity*w->nope;
    const float *q0=w->input+(size_t)hb*hstep;
    const float *q1=q0+w->nope,*q2=q1+w->nope,*q3=q2+w->nope;
    const size_t rstep=(size_t)w->capacity*w->rope;
    const float *r0=w->rope?w->input_rope+(size_t)hb*rstep:q0;
    const float *r1=w->rope?r0+w->rope:q1,*r2=w->rope?r1+w->rope:q2,*r3=w->rope?r2+w->rope:q3;
    float *o0=w->out+(size_t)hb*w->stride;
    float *o1=o0+w->stride,*o2=o1+w->stride,*o3=o2+w->stride;
    for(int key=0;key<total;key+=16) {
      M4 a00={0,0,0,0},a01={0,0,0,0},a02={0,0,0,0},a03={0,0,0,0};
      M4 a10={0,0,0,0},a11={0,0,0,0},a12={0,0,0,0},a13={0,0,0,0};
      M4 a20={0,0,0,0},a21={0,0,0,0},a22={0,0,0,0},a23={0,0,0,0};
      M4 a30={0,0,0,0},a31={0,0,0,0},a32={0,0,0,0},a33={0,0,0,0};
      for(int c=0;c<w->nope;c++) {
        const float *base=w->packed+(size_t)c*pstride+key;
        M4 k0=*(const M4*)(base),k1=*(const M4*)(base+4),k2=*(const M4*)(base+8),k3=*(const M4*)(base+12);
        float v;M4 b;
        v=q0[c];b=(M4){v,v,v,v};a00=a00+b*k0;a01=a01+b*k1;a02=a02+b*k2;a03=a03+b*k3;
        v=q1[c];b=(M4){v,v,v,v};a10=a10+b*k0;a11=a11+b*k1;a12=a12+b*k2;a13=a13+b*k3;
        v=q2[c];b=(M4){v,v,v,v};a20=a20+b*k0;a21=a21+b*k1;a22=a22+b*k2;a23=a23+b*k3;
        v=q3[c];b=(M4){v,v,v,v};a30=a30+b*k0;a31=a31+b*k1;a32=a32+b*k2;a33=a33+b*k3;
      }
      for(int c=0;c<w->rope;c++) {
        const float *base=w->packed+(size_t)(c+w->nope)*pstride+key;
        M4 k0=*(const M4*)(base),k1=*(const M4*)(base+4),k2=*(const M4*)(base+8),k3=*(const M4*)(base+12);
        float v;M4 b;
        v=r0[c];b=(M4){v,v,v,v};a00=a00+b*k0;a01=a01+b*k1;a02=a02+b*k2;a03=a03+b*k3;
        v=r1[c];b=(M4){v,v,v,v};a10=a10+b*k0;a11=a11+b*k1;a12=a12+b*k2;a13=a13+b*k3;
        v=r2[c];b=(M4){v,v,v,v};a20=a20+b*k0;a21=a21+b*k1;a22=a22+b*k2;a23=a23+b*k3;
        v=r3[c];b=(M4){v,v,v,v};a30=a30+b*k0;a31=a31+b*k1;a32=a32+b*k2;a33=a33+b*k3;
      }
      for(int lane=0;lane<4&&key+lane<total;lane++){o0[key+lane]=a00[lane]*w->scale;o1[key+lane]=a10[lane]*w->scale;o2[key+lane]=a20[lane]*w->scale;o3[key+lane]=a30[lane]*w->scale;}
      for(int lane=0;lane<4&&key+4+lane<total;lane++){o0[key+4+lane]=a01[lane]*w->scale;o1[key+4+lane]=a11[lane]*w->scale;o2[key+4+lane]=a21[lane]*w->scale;o3[key+4+lane]=a31[lane]*w->scale;}
      for(int lane=0;lane<4&&key+8+lane<total;lane++){o0[key+8+lane]=a02[lane]*w->scale;o1[key+8+lane]=a12[lane]*w->scale;o2[key+8+lane]=a22[lane]*w->scale;o3[key+8+lane]=a32[lane]*w->scale;}
      for(int lane=0;lane<4&&key+12+lane<total;lane++){o0[key+12+lane]=a03[lane]*w->scale;o1[key+12+lane]=a13[lane]*w->scale;o2[key+12+lane]=a23[lane]*w->scale;o3[key+12+lane]=a33[lane]*w->scale;}
    }
  }
}
static void mla_pv_single(MLAWork *w,int head) {
  const int total=w->start+w->rows;
  const float *p0=w->input+((size_t)head*w->capacity)*w->stride;
  float *out=w->out+((size_t)head*w->capacity)*w->values;
  for(int c=0;c<w->values;c+=4) {
    M4 a={0,0,0,0};
    for(int key=0;key<total;key++) {
      M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
      float p=p0[key];a=a+(M4){p,p,p,p}*value;
    }
    *(M4*)(out+c)=a;
  }
}
static void mla_pv_part_hb(MLAWork *w,int group,int groups) {
  const int HB=MLA_HB;const int total=w->start+w->rows;const int nblocks=(w->heads+HB-1)/HB;
  for(int blk=group;blk<nblocks;blk+=groups) {
    int hb=blk*HB,nh=w->heads-hb;if(nh>HB)nh=HB;
    if(nh<HB){for(int h=0;h<nh;h++)mla_pv_single(w,hb+h);continue;}
    const float *p0=w->input+(size_t)hb*w->capacity*w->stride;
    const float *p1=p0+w->stride,*p2=p1+w->stride,*p3=p2+w->stride;
    float *o0=w->out+(size_t)hb*w->capacity*w->values;
    float *o1=o0+w->values,*o2=o1+w->values,*o3=o2+w->values;
    int c=0;
    for(;c+8<=w->values;c+=8) {
      M4 a00={0,0,0,0},a01={0,0,0,0},a10={0,0,0,0},a11={0,0,0,0};
      M4 a20={0,0,0,0},a21={0,0,0,0},a30={0,0,0,0},a31={0,0,0,0};
      for(int key=0;key<total;key++) {
        const float *cv=w->cache+(size_t)key*w->width+c;
        M4 v0=*(const M4*)cv,v1=*(const M4*)(cv+4);
        float p;M4 b;
        p=p0[key];b=(M4){p,p,p,p};a00=a00+b*v0;a01=a01+b*v1;
        p=p1[key];b=(M4){p,p,p,p};a10=a10+b*v0;a11=a11+b*v1;
        p=p2[key];b=(M4){p,p,p,p};a20=a20+b*v0;a21=a21+b*v1;
        p=p3[key];b=(M4){p,p,p,p};a30=a30+b*v0;a31=a31+b*v1;
      }
      *(M4*)(o0+c)=a00;*(M4*)(o0+c+4)=a01;
      *(M4*)(o1+c)=a10;*(M4*)(o1+c+4)=a11;
      *(M4*)(o2+c)=a20;*(M4*)(o2+c+4)=a21;
      *(M4*)(o3+c)=a30;*(M4*)(o3+c+4)=a31;
    }
    for(;c<w->values;c+=4) {
      M4 a0={0,0,0,0},a1={0,0,0,0},a2={0,0,0,0},a3={0,0,0,0};
      for(int key=0;key<total;key++) {
        M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
        float p;
        p=p0[key];a0=a0+(M4){p,p,p,p}*value;
        p=p1[key];a1=a1+(M4){p,p,p,p}*value;
        p=p2[key];a2=a2+(M4){p,p,p,p}*value;
        p=p3[key];a3=a3+(M4){p,p,p,p}*value;
      }
      *(M4*)(o0+c)=a0;*(M4*)(o1+c)=a1;*(M4*)(o2+c)=a2;*(M4*)(o3+c)=a3;
    }
  }
}
#endif
static void mla_qk_worker(void **args,int unused,int group) {
  (void)unused;MLAWork *w=args[0];if(group<0||group>=4)return;
#if MLA_HEAD_BLOCK > 1
  if(w->capacity==1&&w->rows==1)mla_qk_part_hb(w,group,4);else
#endif
  mla_qk_part(w,group,4);w->done[group]=1;
}
static void mla_pv_worker(void **args,int unused,int group) {
  (void)unused;MLAWork *w=args[0];if(group<0||group>=4)return;
#if MLA_HEAD_BLOCK > 1 && !MLA_FP16
  if(w->capacity==1&&w->rows==1)mla_pv_part_hb(w,group,4);else
#endif
  mla_pv_part(w,group,4);w->done[group]=1;
}
static void mla_execute(MLAWork *w,mla_parallel_fn parallel,int qk) {
  if(w->start<0||w->rows<1||w->rows>w->capacity||w->start+w->rows>w->context)__builtin_trap();
  /* Single-row decode keeps its existing work and avoids the safety scan. */
  w->causal_safe=MLA_CAUSAL_WORK&&w->capacity>1&&w->scale>0.0f&&w->scale<=1024.0f&&
    mla_causal_safe_values(w->cache,(size_t)(w->start+w->rows)*w->width);
  if(!parallel){
#if MLA_HEAD_BLOCK > 1
    if(w->capacity==1&&w->rows==1){if(qk)mla_qk_part_hb(w,0,1);else mla_pv_part_hb(w,0,1);return;}
#endif
    if(qk)mla_qk_part(w,0,1);else mla_pv_part(w,0,1);return;}
  void *args[4]={w,0,0,0};
  if(parallel(qk?mla_qk_worker:mla_pv_worker,args,0))__builtin_trap();
  for(int i=0;i<4;i++)if(!w->done[i])__builtin_trap();
}
