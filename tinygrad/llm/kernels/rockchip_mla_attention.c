/* FP32 MLA dots: independent output lanes, sequential reduction order. */
#include <stddef.h>
#ifndef MLA_CAUSAL_WORK
#define MLA_CAUSAL_WORK 1
#endif
#ifndef MLA_ATTN_ROWS
#define MLA_ATTN_ROWS 0
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
  int fused_pv;
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

/* Multi-row (prefill) head-blocked variants: 4 heads share one K-block load
   (qk) or one latent-cache load (pv). The per-row causal shortcut, the
   -inf masking and the signed-zero/non-finite tail guard are reproduced exactly,
   so each output keeps the reference kernel's bit pattern. */
static void mla_qk_single_rows(MLAWork *w,int head) {
  const int total=w->start+w->rows;const int pstride=((w->context+15)&~15)+4;
  for(int row=0;row<w->rows;row++) {
    const float *q=w->input+((size_t)head*w->capacity+row)*w->nope;
    const float *qr=w->rope?w->input_rope+((size_t)head*w->capacity+row)*w->rope:q;
    float *out=w->out+((size_t)head*w->capacity+row)*w->stride;
    int end=total;
    if(w->causal_safe&&mla_causal_safe_values(q,w->nope)&&mla_causal_safe_values(qr,w->rope))end=w->start+row+1;
    for(int key=0;key<end;key+=16) {
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
      for(int lane=0;lane<4&&key+lane<total;lane++)out[key+lane]=s0[lane]*w->scale+(key+lane>w->start+row?-__builtin_inff():0.0f);
      for(int lane=0;lane<4&&key+4+lane<total;lane++)out[key+4+lane]=s1[lane]*w->scale+(key+4+lane>w->start+row?-__builtin_inff():0.0f);
      for(int lane=0;lane<4&&key+8+lane<total;lane++)out[key+8+lane]=s2[lane]*w->scale+(key+8+lane>w->start+row?-__builtin_inff():0.0f);
      for(int lane=0;lane<4&&key+12+lane<total;lane++)out[key+12+lane]=s3[lane]*w->scale+(key+12+lane>w->start+row?-__builtin_inff():0.0f);
    }
    for(int key=(end+15)&~15;key<total;key++)out[key]=-__builtin_inff();
  }
}
static void mla_qk_part_hb_rows(MLAWork *w,int group,int groups) {
  const int HB=MLA_HB;const int total=w->start+w->rows;const int pstride=((w->context+15)&~15)+4;
  const int nblocks=(w->heads+HB-1)/HB;
  for(int blk=group;blk<nblocks;blk+=groups) {
    int hb=blk*HB,nh=w->heads-hb;if(nh>HB)nh=HB;
    if(nh<HB){for(int h=0;h<nh;h++)mla_qk_single_rows(w,hb+h);continue;}
    for(int row=0;row<w->rows;row++) {
      const size_t qstep=(size_t)w->capacity*w->nope,rstep=(size_t)w->capacity*w->rope;
      const float *q0=w->input+((size_t)hb*w->capacity+row)*w->nope;
      const float *q1=q0+qstep,*q2=q1+qstep,*q3=q2+qstep;
      const float *r0=w->rope?w->input_rope+((size_t)hb*w->capacity+row)*w->rope:q0;
      const float *r1=w->rope?r0+rstep:q1,*r2=w->rope?r1+rstep:q2,*r3=w->rope?r2+rstep:q3;
      int end=total;
      if(w->causal_safe&&mla_causal_safe_values(q0,w->nope)&&mla_causal_safe_values(q1,w->nope)
         &&mla_causal_safe_values(q2,w->nope)&&mla_causal_safe_values(q3,w->nope)
         &&(w->rope==0||(mla_causal_safe_values(r0,w->rope)&&mla_causal_safe_values(r1,w->rope)
                         &&mla_causal_safe_values(r2,w->rope)&&mla_causal_safe_values(r3,w->rope))))
        end=w->start+row+1;
      for(int key=0;key<end;key+=16) {
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
        /* write with the same mask expression as the reference kernels */
        float *o0=w->out+((size_t)(hb+0)*w->capacity+row)*w->stride;
        float *o1=w->out+((size_t)(hb+1)*w->capacity+row)*w->stride;
        float *o2=w->out+((size_t)(hb+2)*w->capacity+row)*w->stride;
        float *o3=w->out+((size_t)(hb+3)*w->capacity+row)*w->stride;
        for(int lane=0;lane<4&&key+lane<total;lane++){int idx=key+lane;float m=idx>w->start+row?-__builtin_inff():0.0f;o0[idx]=a00[lane]*w->scale+m;o1[idx]=a10[lane]*w->scale+m;o2[idx]=a20[lane]*w->scale+m;o3[idx]=a30[lane]*w->scale+m;}
        for(int lane=0;lane<4&&key+4+lane<total;lane++){int idx=key+4+lane;float m=idx>w->start+row?-__builtin_inff():0.0f;o0[idx]=a01[lane]*w->scale+m;o1[idx]=a11[lane]*w->scale+m;o2[idx]=a21[lane]*w->scale+m;o3[idx]=a31[lane]*w->scale+m;}
        for(int lane=0;lane<4&&key+8+lane<total;lane++){int idx=key+8+lane;float m=idx>w->start+row?-__builtin_inff():0.0f;o0[idx]=a02[lane]*w->scale+m;o1[idx]=a12[lane]*w->scale+m;o2[idx]=a22[lane]*w->scale+m;o3[idx]=a32[lane]*w->scale+m;}
        for(int lane=0;lane<4&&key+12+lane<total;lane++){int idx=key+12+lane;float m=idx>w->start+row?-__builtin_inff():0.0f;o0[idx]=a03[lane]*w->scale+m;o1[idx]=a13[lane]*w->scale+m;o2[idx]=a23[lane]*w->scale+m;o3[idx]=a33[lane]*w->scale+m;}
      }
      for(int key=(end+15)&~15;key<total;key++) {
        float *o0=w->out+((size_t)(hb+0)*w->capacity+row)*w->stride;
        float *o1=w->out+((size_t)(hb+1)*w->capacity+row)*w->stride;
        float *o2=w->out+((size_t)(hb+2)*w->capacity+row)*w->stride;
        float *o3=w->out+((size_t)(hb+3)*w->capacity+row)*w->stride;
        o0[key]=o1[key]=o2[key]=o3[key]=-__builtin_inff();
      }
    }
  }
}
static void mla_pv_single_rows(MLAWork *w,int head) {
  const int total=w->start+w->rows;
  const float *ph=w->input+(size_t)head*w->capacity*w->stride;
  float *oh=w->out+(size_t)head*w->capacity*w->values;
  for(int row=0;row<w->rows;row++) {
    const float *p0=ph+(size_t)row*w->stride;
    float *out=oh+(size_t)row*w->values;
    int end=total;
    if(w->causal_safe&&w->start+row+1<total) {
      int candidate=w->start+row+1,zero_tail=1;
      for(int key=candidate;key<total;key++)if(p0[key]!=0.0f){zero_tail=0;break;}
      if(zero_tail)end=candidate;
    }
    for(int c=0;c<w->values;c+=4) {
      M4 a={0,0,0,0};
      for(int key=0;key<end;key++) {
        M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
        float p=p0[key];a=a+(M4){p,p,p,p}*value;
      }
      if(end<total&&mla_tail_needed(a)) {
        for(int key=end;key<total;key++) {
          M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
          float p=p0[key];a=a+(M4){p,p,p,p}*value;
        }
      }
      *(M4*)(out+c)=a;
    }
  }
}
/* One shared implementation of the four-head weighted-value row: both the
   reference pv kernel and the fused qk+softmax+value kernel call it, so the
   floating-point operation order is identical by construction. */
static __attribute__((noinline)) void mla_pv_row4(MLAWork *w,const float *p0,const float *p1,
    const float *p2,const float *p3,float *o0,float *o1,float *o2,float *o3,int row) {
  const int total=w->start+w->rows;
  int end=total;
  if(w->causal_safe&&w->start+row+1<total) {
    int candidate=w->start+row+1,zero_tail=1;
    for(int key=candidate;key<total&&zero_tail;key++) {
      if(p0[key]!=0.0f||p1[key]!=0.0f||p2[key]!=0.0f||p3[key]!=0.0f)zero_tail=0;
    }
    if(zero_tail)end=candidate;
  }
  for(int c=0;c<w->values;c+=4) {
    M4 a0={0,0,0,0},a1={0,0,0,0},a2={0,0,0,0},a3={0,0,0,0};
    for(int key=0;key<end;key++) {
      M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
      float p;M4 b;
      p=p0[key];b=(M4){p,p,p,p};a0=a0+b*value;
      p=p1[key];b=(M4){p,p,p,p};a1=a1+b*value;
      p=p2[key];b=(M4){p,p,p,p};a2=a2+b*value;
      p=p3[key];b=(M4){p,p,p,p};a3=a3+b*value;
    }
    if(end<total&&(mla_tail_needed(a0)||mla_tail_needed(a1)||mla_tail_needed(a2)||mla_tail_needed(a3))) {
      for(int key=end;key<total;key++) {
        M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
        float p;M4 b;
        p=p0[key];b=(M4){p,p,p,p};a0=a0+b*value;
        p=p1[key];b=(M4){p,p,p,p};a1=a1+b*value;
        p=p2[key];b=(M4){p,p,p,p};a2=a2+b*value;
        p=p3[key];b=(M4){p,p,p,p};a3=a3+b*value;
      }
    }
    *(M4*)(o0+c)=a0;*(M4*)(o1+c)=a1;*(M4*)(o2+c)=a2;*(M4*)(o3+c)=a3;
  }
}
static void mla_pv_part_hb_rows(MLAWork *w,int group,int groups) {
  const int HB=MLA_HB;const int total=w->start+w->rows;
  const int nblocks=(w->heads+HB-1)/HB;
  for(int blk=group;blk<nblocks;blk+=groups) {
    int hb=blk*HB,nh=w->heads-hb;if(nh>HB)nh=HB;
    if(nh<HB){for(int h=0;h<nh;h++)mla_pv_single_rows(w,hb+h);continue;}
    const float *ph0=w->input+(size_t)(hb+0)*w->capacity*w->stride;
    const float *ph1=w->input+(size_t)(hb+1)*w->capacity*w->stride;
    const float *ph2=w->input+(size_t)(hb+2)*w->capacity*w->stride;
    const float *ph3=w->input+(size_t)(hb+3)*w->capacity*w->stride;
    float *oh0=w->out+(size_t)(hb+0)*w->capacity*w->values;
    float *oh1=w->out+(size_t)(hb+1)*w->capacity*w->values;
    float *oh2=w->out+(size_t)(hb+2)*w->capacity*w->values;
    float *oh3=w->out+(size_t)(hb+3)*w->capacity*w->values;
    for(int row=0;row<w->rows;row++) mla_pv_row4(w,
      ph0+(size_t)row*w->stride,ph1+(size_t)row*w->stride,ph2+(size_t)row*w->stride,ph3+(size_t)row*w->stride,
      oh0+(size_t)row*w->values,oh1+(size_t)row*w->values,oh2+(size_t)row*w->values,oh3+(size_t)row*w->values,row);
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

#ifdef MLA_FUSED_QK_SOFTMAX
/* Fused qk + exact softmax (prefill): the per-row scores stay in a small local
   buffer, so the padded FP32 score buffer is never written or read. The qk
   accumulation (causal end, ascending c), the -inf causal mask, the sequential
   max/sum and the vector exp2 polynomial are reproduced exactly, so the
   probabilities are bit-identical to running the two reference kernels. */
static void mla_qk_softmax_part_ref(MLAWork *w,int group,int groups,int pv);
#if MLA_ATTN_ROWS >= 2
/* Row-paired fused qk+softmax+pv for a 2-head block: two rows share each K tile
   (16 keys = 4 M4) with 2*2*4 = 16 M4 accumulators, so the instruction mix per
   output is identical to the shipped 4-head/1-row kernel while the number of
   equal-cost work units doubles (20 blocks -> 5/5/5/5 over the 4 workers, no
   33% imbalance). Per-output FP order is unchanged -> bit-identical. */
static __attribute__((noinline)) void mla_pv_row2x2(MLAWork *w,
    const float *pa0,const float *pa1,const float *pb0,const float *pb1,
    float *oa0,float *oa1,float *ob0,float *ob1,int row) {
  const int total=w->start+w->rows;
  int enda=total,endb=total;
  if(w->causal_safe&&w->start+row+1<total) {
    int candidate=w->start+row+1,zero_tail=1;
    for(int key=candidate;key<total&&zero_tail;key++) if(pa0[key]!=0.0f||pa1[key]!=0.0f)zero_tail=0;
    if(zero_tail)enda=candidate;
  }
  if(w->causal_safe&&w->start+row+2<total) {
    int candidate=w->start+row+2,zero_tail=1;
    for(int key=candidate;key<total&&zero_tail;key++) if(pb0[key]!=0.0f||pb1[key]!=0.0f)zero_tail=0;
    if(zero_tail)endb=candidate;
  }
  const int shared=enda<endb?enda:endb;
  for(int c=0;c<w->values;c+=4) {
    M4 aa0={0,0,0,0},aa1={0,0,0,0},ba0={0,0,0,0},ba1={0,0,0,0};
    int key=0;
    for(;key<shared;key++) {
      M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
      float p;M4 b;
      p=pa0[key];b=(M4){p,p,p,p};aa0=aa0+b*value;
      p=pa1[key];b=(M4){p,p,p,p};aa1=aa1+b*value;
      p=pb0[key];b=(M4){p,p,p,p};ba0=ba0+b*value;
      p=pb1[key];b=(M4){p,p,p,p};ba1=ba1+b*value;
    }
    for(;key<enda;key++) {
      M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
      float p;M4 b;
      p=pa0[key];b=(M4){p,p,p,p};aa0=aa0+b*value;
      p=pa1[key];b=(M4){p,p,p,p};aa1=aa1+b*value;
    }
    for(;key<endb;key++) {
      M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
      float p;M4 b;
      p=pb0[key];b=(M4){p,p,p,p};ba0=ba0+b*value;
      p=pb1[key];b=(M4){p,p,p,p};ba1=ba1+b*value;
    }
    if(enda<total&&(mla_tail_needed(aa0)||mla_tail_needed(aa1))) {
      for(int key2=enda;key2<total;key2++) {
        M4 value=*(const M4*)(w->cache+(size_t)key2*w->width+c);
        float p;M4 b;
        p=pa0[key2];b=(M4){p,p,p,p};aa0=aa0+b*value;
        p=pa1[key2];b=(M4){p,p,p,p};aa1=aa1+b*value;
      }
    }
    if(endb<total&&(mla_tail_needed(ba0)||mla_tail_needed(ba1))) {
      for(int key2=endb;key2<total;key2++) {
        M4 value=*(const M4*)(w->cache+(size_t)key2*w->width+c);
        float p;M4 b;
        p=pb0[key2];b=(M4){p,p,p,p};ba0=ba0+b*value;
        p=pb1[key2];b=(M4){p,p,p,p};ba1=ba1+b*value;
      }
    }
    *(M4*)(oa0+c)=aa0;*(M4*)(oa1+c)=aa1;*(M4*)(ob0+c)=ba0;*(M4*)(ob1+c)=ba1;
  }
}
static void mla_qk_softmax_part_rows2(MLAWork *w,int group,int groups,int pv) {
  const int total=w->start+w->rows;
  const int pstride=((w->context+15)&~15)+4;
  float sc[2][2][1024];
  const int nblocks=w->heads/2;
  const size_t qstep=(size_t)w->capacity*w->nope,rstep=(size_t)w->capacity*w->rope;
  for(int blk=group;blk<nblocks;blk+=groups) {
    const int hb=blk*2;
    for(int row=0;row<w->rows;row+=2) {
      const float *p00=w->input+((size_t)(hb+0)*w->capacity+row)*w->nope;
      const float *p10=p00+qstep;
      const float *p01=w->input+((size_t)(hb+0)*w->capacity+row+1)*w->nope;
      const float *p11=p01+qstep;
      const float *r00=w->rope?w->input_rope+((size_t)(hb+0)*w->capacity+row)*w->rope:p00;
      const float *r10=w->rope?r00+rstep:p10;
      const float *s00=w->rope?w->input_rope+((size_t)(hb+0)*w->capacity+row+1)*w->rope:p01;
      const float *s10=w->rope?s00+rstep:p11;
      int end0=total,end1=total;
      if(w->causal_safe&&mla_causal_safe_values(p00,w->nope)&&mla_causal_safe_values(p10,w->nope)
         &&(w->rope==0||(mla_causal_safe_values(r00,w->rope)&&mla_causal_safe_values(r10,w->rope))))
        end0=w->start+row+1;
      if(w->causal_safe&&mla_causal_safe_values(p01,w->nope)&&mla_causal_safe_values(p11,w->nope)
         &&(w->rope==0||(mla_causal_safe_values(s00,w->rope)&&mla_causal_safe_values(s10,w->rope))))
        end1=w->start+row+2;
      const int kend=end0>end1?end0:end1;
      for(int key=0;key<kend;key+=16) {
        M4 a00={0,0,0,0},a01={0,0,0,0},a02={0,0,0,0},a03={0,0,0,0};
        M4 a10={0,0,0,0},a11={0,0,0,0},a12={0,0,0,0},a13={0,0,0,0};
        M4 b00={0,0,0,0},b01={0,0,0,0},b02={0,0,0,0},b03={0,0,0,0};
        M4 b10={0,0,0,0},b11={0,0,0,0},b12={0,0,0,0},b13={0,0,0,0};
        for(int c=0;c<w->nope;c++) {
          const float *base=w->packed+(size_t)c*pstride+key;
          M4 k0=*(const M4*)(base),k1=*(const M4*)(base+4),k2=*(const M4*)(base+8),k3=*(const M4*)(base+12);
          float v;M4 t;
          v=p00[c];t=(M4){v,v,v,v};a00=a00+t*k0;a01=a01+t*k1;a02=a02+t*k2;a03=a03+t*k3;
          v=p10[c];t=(M4){v,v,v,v};a10=a10+t*k0;a11=a11+t*k1;a12=a12+t*k2;a13=a13+t*k3;
          v=p01[c];t=(M4){v,v,v,v};b00=b00+t*k0;b01=b01+t*k1;b02=b02+t*k2;b03=b03+t*k3;
          v=p11[c];t=(M4){v,v,v,v};b10=b10+t*k0;b11=b11+t*k1;b12=b12+t*k2;b13=b13+t*k3;
        }
        for(int c=0;c<w->rope;c++) {
          const float *base=w->packed+(size_t)(c+w->nope)*pstride+key;
          M4 k0=*(const M4*)(base),k1=*(const M4*)(base+4),k2=*(const M4*)(base+8),k3=*(const M4*)(base+12);
          float v;M4 t;
          v=r00[c];t=(M4){v,v,v,v};a00=a00+t*k0;a01=a01+t*k1;a02=a02+t*k2;a03=a03+t*k3;
          v=r10[c];t=(M4){v,v,v,v};a10=a10+t*k0;a11=a11+t*k1;a12=a12+t*k2;a13=a13+t*k3;
          v=s00[c];t=(M4){v,v,v,v};b00=b00+t*k0;b01=b01+t*k1;b02=b02+t*k2;b03=b03+t*k3;
          v=s10[c];t=(M4){v,v,v,v};b10=b10+t*k0;b11=b11+t*k1;b12=b12+t*k2;b13=b13+t*k3;
        }
        for(int lane=0;lane<4;lane++){int idx=key+lane;if(idx<total){
          sc[0][0][idx]=idx<end0?a00[lane]*w->scale:-__builtin_inff();
          sc[0][1][idx]=idx<end0?a10[lane]*w->scale:-__builtin_inff();
          sc[1][0][idx]=idx<end1?b00[lane]*w->scale:-__builtin_inff();
          sc[1][1][idx]=idx<end1?b10[lane]*w->scale:-__builtin_inff();}}
        for(int lane=0;lane<4;lane++){int idx=key+4+lane;if(idx<total){
          sc[0][0][idx]=idx<end0?a01[lane]*w->scale:-__builtin_inff();
          sc[0][1][idx]=idx<end0?a11[lane]*w->scale:-__builtin_inff();
          sc[1][0][idx]=idx<end1?b01[lane]*w->scale:-__builtin_inff();
          sc[1][1][idx]=idx<end1?b11[lane]*w->scale:-__builtin_inff();}}
        for(int lane=0;lane<4;lane++){int idx=key+8+lane;if(idx<total){
          sc[0][0][idx]=idx<end0?a02[lane]*w->scale:-__builtin_inff();
          sc[0][1][idx]=idx<end0?a12[lane]*w->scale:-__builtin_inff();
          sc[1][0][idx]=idx<end1?b02[lane]*w->scale:-__builtin_inff();
          sc[1][1][idx]=idx<end1?b12[lane]*w->scale:-__builtin_inff();}}
        for(int lane=0;lane<4;lane++){int idx=key+12+lane;if(idx<total){
          sc[0][0][idx]=idx<end0?a03[lane]*w->scale:-__builtin_inff();
          sc[0][1][idx]=idx<end0?a13[lane]*w->scale:-__builtin_inff();
          sc[1][0][idx]=idx<end1?b03[lane]*w->scale:-__builtin_inff();
          sc[1][1][idx]=idx<end1?b13[lane]*w->scale:-__builtin_inff();}}
      }
      for(int h=0;h<2;h++) {
        for(int key=end0;key<total;key++)sc[0][h][key]=-__builtin_inff();
        for(int key=end1;key<total;key++)sc[1][h][key]=-__builtin_inff();
      }
      for(int r=0;r<2;r++)for(int h=0;h<2;h++) {
        float maximum=-__builtin_inff();
        for(int k=0;k<total;k++)if(maximum<sc[r][h][k])maximum=sc[r][h][k];
        float sum=0.0f;
        int k=0;
        for(;k+4<=total;k+=4) {
          float4 x={sc[r][h][k],sc[r][h][k+1],sc[r][h][k+2],sc[r][h][k+3]};
          x=(x-(float4)maximum)*(float4)1.4426950216293335f;
          float4 e=rockchip_exp2_vector(x);
          sc[r][h][k]=e[0];sc[r][h][k+1]=e[1];sc[r][h][k+2]=e[2];sc[r][h][k+3]=e[3];
          sum+=e[0];sum+=e[1];sum+=e[2];sum+=e[3];
        }
        for(;k<total;k++) {
          float x=(sc[r][h][k]-maximum)*1.4426950216293335f;
          float4 e4=rockchip_exp2_vector((float4){x,x,x,x});
          sc[r][h][k]=e4[0];sum+=e4[0];
        }
        for(k=0;k<total;k++)sc[r][h][k]/=sum;
      }
      if(pv) {
        mla_pv_row2x2(w,sc[0][0],sc[0][1],sc[1][0],sc[1][1],
          w->out+((size_t)(hb+0)*w->capacity+row)*w->values,w->out+((size_t)(hb+1)*w->capacity+row)*w->values,
          w->out+((size_t)(hb+0)*w->capacity+row+1)*w->values,w->out+((size_t)(hb+1)*w->capacity+row+1)*w->values,row);
      } else {
        for(int r=0;r<2;r++)for(int h=0;h<2;h++) {
          float *po=w->out+((size_t)(hb+h)*w->capacity+row+r)*w->stride;
          int k=0;
          for(;k<total;k++)po[k]=sc[r][h][k];
          for(;k<w->stride;k++)po[k]=0.0f;
        }
      }
    }
  }
}
static void mla_qk_softmax_part(MLAWork *w,int group,int groups,int pv) {
  if((w->heads&1)==0&&(w->rows&3)==0&&MLA_HEAD_BLOCK==4) { mla_qk_softmax_part_rows2(w,group,groups,pv); return; }
  mla_qk_softmax_part_ref(w,group,groups,pv);
}
#else
static void mla_qk_softmax_part(MLAWork *w,int group,int groups,int pv) {
  mla_qk_softmax_part_ref(w,group,groups,pv);
}
#endif
static void mla_qk_softmax_part_ref(MLAWork *w,int group,int groups,int pv) {
  const int total=w->start+w->rows;
  const int pstride=((w->context+15)&~15)+4;
  float sc[4][1024];
  const int HB=MLA_HEAD_BLOCK>1?MLA_HEAD_BLOCK:1;
  const int nblocks=(w->heads+HB-1)/HB;
  for(int blk=group;blk<nblocks;blk+=groups) {
    int hb=blk*HB,nh=w->heads-hb;if(nh>HB)nh=HB;
    for(int row=0;row<w->rows;row++) {
      const float *q0,*q1,*q2,*q3,*r0,*r1,*r2,*r3;
      q0=w->input+((size_t)(hb+0)*w->capacity+row)*w->nope;
      q1=nh>1?q0+(size_t)w->capacity*w->nope:q0;
      q2=nh>2?q1+(size_t)w->capacity*w->nope:q0;
      q3=nh>3?q2+(size_t)w->capacity*w->nope:q0;
      r0=w->rope?w->input_rope+((size_t)(hb+0)*w->capacity+row)*w->rope:q0;
      r1=w->rope?(nh>1?r0+(size_t)w->capacity*w->rope:r0):q0;
      r2=w->rope?(nh>2?r1+(size_t)w->capacity*w->rope:r0):q0;
      r3=w->rope?(nh>3?r2+(size_t)w->capacity*w->rope:r0):q0;
      int end=total;
      if(w->causal_safe&&mla_causal_safe_values(q0,w->nope)&&mla_causal_safe_values(q1,w->nope)
         &&mla_causal_safe_values(q2,w->nope)&&mla_causal_safe_values(q3,w->nope)
         &&(w->rope==0||(mla_causal_safe_values(r0,w->rope)&&mla_causal_safe_values(r1,w->rope)
                         &&mla_causal_safe_values(r2,w->rope)&&mla_causal_safe_values(r3,w->rope))))
        end=w->start+row+1;
      for(int key=0;key<end;key+=16) {
        M4 a00={0,0,0,0},a01={0,0,0,0},a02={0,0,0,0},a03={0,0,0,0};
        M4 a10={0,0,0,0},a11={0,0,0,0},a12={0,0,0,0},a13={0,0,0,0};
        M4 a20={0,0,0,0},a21={0,0,0,0},a22={0,0,0,0},a23={0,0,0,0};
        M4 a30={0,0,0,0},a31={0,0,0,0},a32={0,0,0,0},a33={0,0,0,0};
        for(int c=0;c<w->nope;c++) {
          const float *base=w->packed+(size_t)c*pstride+key;
          M4 k0=*(const M4*)(base),k1=*(const M4*)(base+4),k2=*(const M4*)(base+8),k3=*(const M4*)(base+12);
          float v;M4 b;
          v=q0[c];b=(M4){v,v,v,v};a00=a00+b*k0;a01=a01+b*k1;a02=a02+b*k2;a03=a03+b*k3;
          if(nh>1){v=q1[c];b=(M4){v,v,v,v};a10=a10+b*k0;a11=a11+b*k1;a12=a12+b*k2;a13=a13+b*k3;}
          if(nh>2){v=q2[c];b=(M4){v,v,v,v};a20=a20+b*k0;a21=a21+b*k1;a22=a22+b*k2;a23=a23+b*k3;}
          if(nh>3){v=q3[c];b=(M4){v,v,v,v};a30=a30+b*k0;a31=a31+b*k1;a32=a32+b*k2;a33=a33+b*k3;}
        }
        for(int c=0;c<w->rope;c++) {
          const float *base=w->packed+(size_t)(c+w->nope)*pstride+key;
          M4 k0=*(const M4*)(base),k1=*(const M4*)(base+4),k2=*(const M4*)(base+8),k3=*(const M4*)(base+12);
          float v;M4 b;
          v=r0[c];b=(M4){v,v,v,v};a00=a00+b*k0;a01=a01+b*k1;a02=a02+b*k2;a03=a03+b*k3;
          if(nh>1){v=r1[c];b=(M4){v,v,v,v};a10=a10+b*k0;a11=a11+b*k1;a12=a12+b*k2;a13=a13+b*k3;}
          if(nh>2){v=r2[c];b=(M4){v,v,v,v};a20=a20+b*k0;a21=a21+b*k1;a22=a22+b*k2;a23=a23+b*k3;}
          if(nh>3){v=r3[c];b=(M4){v,v,v,v};a30=a30+b*k0;a31=a31+b*k1;a32=a32+b*k2;a33=a33+b*k3;}
        }
        for(int h=0;h<nh;h++) {
          M4 *aa = h==0?&a00:h==1?&a10:h==2?&a20:&a30;
          M4 *ab = h==0?&a01:h==1?&a11:h==2?&a21:&a31;
          M4 *ac = h==0?&a02:h==1?&a12:h==2?&a22:&a32;
          M4 *ad = h==0?&a03:h==1?&a13:h==2?&a23:&a33;
          for(int lane=0;lane<4;lane++){int idx=key+lane;
            if(idx<total)sc[h][idx]=idx<end?(*aa)[lane]*w->scale:-__builtin_inff();}
          for(int lane=0;lane<4;lane++){int idx=key+4+lane;
            if(idx<total)sc[h][idx]=idx<end?(*ab)[lane]*w->scale:-__builtin_inff();}
          for(int lane=0;lane<4;lane++){int idx=key+8+lane;
            if(idx<total)sc[h][idx]=idx<end?(*ac)[lane]*w->scale:-__builtin_inff();}
          for(int lane=0;lane<4;lane++){int idx=key+12+lane;
            if(idx<total)sc[h][idx]=idx<end?(*ad)[lane]*w->scale:-__builtin_inff();}
        }
      }
      for(int key=(end+15)&~15;key<total;key++)for(int h=0;h<nh;h++)sc[h][key]=-__builtin_inff();
      for(int h=0;h<nh;h++) {
        float maximum=-__builtin_inff();
        for(int k=0;k<total;k++)if(maximum<sc[h][k])maximum=sc[h][k];
        float sum=0.0f;
        int k=0;
        for(;k+4<=total;k+=4) {
          float4 x={sc[h][k],sc[h][k+1],sc[h][k+2],sc[h][k+3]};
          x=(x-(float4)maximum)*(float4)1.4426950216293335f;
          float4 e=rockchip_exp2_vector(x);
          sc[h][k]=e[0];sc[h][k+1]=e[1];sc[h][k+2]=e[2];sc[h][k+3]=e[3];
          sum+=e[0];sum+=e[1];sum+=e[2];sum+=e[3];
        }
        for(;k<total;k++) {
          float x=(sc[h][k]-maximum)*1.4426950216293335f;
          float4 e4=rockchip_exp2_vector((float4){x,x,x,x});
          sc[h][k]=e4[0];sum+=e4[0];
        }
        for(k=0;k<total;k++)sc[h][k]/=sum;
        if(!pv) {
          float *po=w->out+((size_t)(hb+h)*w->capacity+row)*w->stride;
          for(k=0;k<total;k++)po[k]=sc[h][k];
          for(;k<w->stride;k++)po[k]=0.0f;
        }
      }
      if(pv&&nh==4) {
        mla_pv_row4(w,sc[0],sc[1],sc[2],sc[3],
          w->out+((size_t)(hb+0)*w->capacity+row)*w->values,
          w->out+((size_t)(hb+1)*w->capacity+row)*w->values,
          w->out+((size_t)(hb+2)*w->capacity+row)*w->values,
          w->out+((size_t)(hb+3)*w->capacity+row)*w->values,row);
      } else if(pv) {
        for(int h=0;h<nh;h++) {
          float *o=w->out+((size_t)(hb+h)*w->capacity+row)*w->values;
          int pvend=total;
          if(w->causal_safe&&w->start+row+1<total) {
            int candidate=w->start+row+1,zero=1;
            for(int key=candidate;key<total;key++)if(sc[h][key]!=0.0f){zero=0;break;}
            if(zero)pvend=candidate;
          }
          for(int c=0;c<w->values;c+=4) {
            M4 a={0,0,0,0};
            for(int key=0;key<pvend;key++) {
              M4 value=*(const M4*)(w->cache+(size_t)key*w->width+c);
              float p=sc[h][key];a=a+(M4){p,p,p,p}*value;
            }
            *(M4*)(o+c)=a;
          }
        }
      }
    }
  }
}
#endif
static void mla_qk_worker(void **args,int unused,int group) {
  (void)unused;MLAWork *w=args[0];if(group<0||group>=4)return;
#if MLA_HEAD_BLOCK > 1
  if(w->capacity==1&&w->rows==1)mla_qk_part_hb(w,group,4);else mla_qk_part_hb_rows(w,group,4);
#else
  mla_qk_part(w,group,4);
#endif
  w->done[group]=1;
}
static void mla_pv_worker(void **args,int unused,int group) {
  (void)unused;MLAWork *w=args[0];if(group<0||group>=4)return;
#if MLA_HEAD_BLOCK > 1 && !MLA_FP16
  if(w->capacity==1&&w->rows==1)mla_pv_part_hb(w,group,4);else mla_pv_part_hb_rows(w,group,4);
#else
  mla_pv_part(w,group,4);
#endif
  w->done[group]=1;
}
#ifdef MLA_FUSED_QK_SOFTMAX
static void mla_qk_softmax_worker(void **args,int unused,int group) {
  (void)unused;MLAWork *w=args[0];if(group<0||group>=4)return;
  mla_qk_softmax_part(w,group,4,w->fused_pv);
  w->done[group]=1;
}
static void mla_execute_qk_softmax(MLAWork *w,mla_parallel_fn parallel,int pv) {
  if(w->start<0||w->rows<1||w->rows>w->capacity||w->start+w->rows>w->context)__builtin_trap();
  w->causal_safe=MLA_CAUSAL_WORK&&w->capacity>1&&w->scale>0.0f&&w->scale<=1024.0f&&
    mla_causal_safe_values(w->cache,(size_t)(w->start+w->rows)*w->width);
  w->fused_pv=pv;
  if(!parallel){mla_qk_softmax_part(w,0,1,pv);return;}
  void *args[4]={w,0,0,0};
  if(parallel(mla_qk_softmax_worker,args,0))__builtin_trap();
  for(int i=0;i<4;i++)if(!w->done[i])__builtin_trap();
}
#endif
static void mla_execute(MLAWork *w,mla_parallel_fn parallel,int qk) {
  if(w->start<0||w->rows<1||w->rows>w->capacity||w->start+w->rows>w->context)__builtin_trap();
  /* Single-row decode keeps its existing work and avoids the safety scan. */
  w->causal_safe=MLA_CAUSAL_WORK&&w->capacity>1&&w->scale>0.0f&&w->scale<=1024.0f&&
    mla_causal_safe_values(w->cache,(size_t)(w->start+w->rows)*w->width);
  if(!parallel){
#if MLA_HEAD_BLOCK > 1 && !MLA_FP16
    if(w->capacity==1&&w->rows==1){if(qk)mla_qk_part_hb(w,0,1);else mla_pv_part_hb(w,0,1);return;}
    if(qk)mla_qk_part_hb_rows(w,0,1);else mla_pv_part_hb_rows(w,0,1);return;
#endif
    if(qk)mla_qk_part(w,0,1);else mla_pv_part(w,0,1);return;}
  void *args[4]={w,0,0,0};
  if(parallel(qk?mla_qk_worker:mla_pv_worker,args,0))__builtin_trap();
  for(int i=0;i<4;i++)if(!w->done[i])__builtin_trap();
}
