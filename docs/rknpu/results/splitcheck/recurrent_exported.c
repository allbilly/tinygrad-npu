/* CPU GDN scan with serial key reduction and SIMD across value rows. */
#include <stddef.h>
extern void *memset(void *, int, size_t);
typedef float F4 __attribute__((vector_size(16), aligned(1)));

static F4 load_rows(const float *p, int count) {
  if(count==4)return *(const F4 *)p;
  return (F4){p[0],count>1?p[1]:0,count>2?p[2]:0,0};
}
static void store_rows(float *p, F4 x, int count) {
  if(count==4){*(F4 *)p=x;return;}
  for(int i=0;i<count;i++)p[i]=x[i];
}

/* Exact key-ordered recurrence on four value rows, without whole-state transposes.
   Used only for a single decode row with scalar decay and no history. */
#ifndef ONPU_GDN_DIRECT_DECODE
#define ONPU_GDN_DIRECT_DECODE 0
#endif

static __attribute__((always_inline)) void transpose4(F4 a,F4 b,F4 c,F4 d,F4 *x0,F4 *x1,F4 *x2,F4 *x3) {
  F4 ab0=__builtin_shufflevector(a,b,0,4,1,5),ab1=__builtin_shufflevector(a,b,2,6,3,7);
  F4 cd0=__builtin_shufflevector(c,d,0,4,1,5),cd1=__builtin_shufflevector(c,d,2,6,3,7);
  *x0=__builtin_shufflevector(ab0,cd0,0,1,4,5);*x1=__builtin_shufflevector(ab0,cd0,2,3,6,7);
  *x2=__builtin_shufflevector(ab1,cd1,0,1,4,5);*x3=__builtin_shufflevector(ab1,cd1,2,3,6,7);
}
static void direct_row(float *out,const float *q,const float *k,const float *v,float beta,float alpha,float *s,int key_dim,int value_dim) {
  for(int r=0;r<value_dim;r+=4) {
    float *p0=s+(size_t)r*key_dim,*p1=p0+key_dim,*p2=p1+key_dim,*p3=p2+key_dim;
    F4 predicted={0,0,0,0},decay={alpha,alpha,alpha,alpha};
    for(int c=0;c<key_dim;c+=4) {
      F4 x0,x1,x2,x3;
      transpose4(*(const F4*)(p0+c),*(const F4*)(p1+c),*(const F4*)(p2+c),*(const F4*)(p3+c),&x0,&x1,&x2,&x3);
      predicted=predicted+x0*(F4){k[c],k[c],k[c],k[c]};
      predicted=predicted+x1*(F4){k[c+1],k[c+1],k[c+1],k[c+1]};
      predicted=predicted+x2*(F4){k[c+2],k[c+2],k[c+2],k[c+2]};
      predicted=predicted+x3*(F4){k[c+3],k[c+3],k[c+3],k[c+3]};
    }
    F4 delta=(*(const F4*)(v+r)-predicted*decay)*(F4){beta,beta,beta,beta};
    F4 result={0,0,0,0};
    for(int c=0;c<key_dim;c+=4) {
      F4 x0,x1,x2,x3,a,b,d,e;
      transpose4(*(const F4*)(p0+c),*(const F4*)(p1+c),*(const F4*)(p2+c),*(const F4*)(p3+c),&x0,&x1,&x2,&x3);
      x0=x0*decay+delta*(F4){k[c],k[c],k[c],k[c]};
      x1=x1*decay+delta*(F4){k[c+1],k[c+1],k[c+1],k[c+1]};
      x2=x2*decay+delta*(F4){k[c+2],k[c+2],k[c+2],k[c+2]};
      x3=x3*decay+delta*(F4){k[c+3],k[c+3],k[c+3],k[c+3]};
      result=result+x0*(F4){q[c],q[c],q[c],q[c]};
      result=result+x1*(F4){q[c+1],q[c+1],q[c+1],q[c+1]};
      result=result+x2*(F4){q[c+2],q[c+2],q[c+2],q[c+2]};
      result=result+x3*(F4){q[c+3],q[c+3],q[c+3],q[c+3]};
      transpose4(x0,x1,x2,x3,&a,&b,&d,&e);
      *(F4*)(p0+c)=a;*(F4*)(p1+c)=b;*(F4*)(p2+c)=d;*(F4*)(p3+c)=e;
    }
    *(F4*)(out+r)=result;
  }
}

static int scan_disjoint(const void *a,size_t a_bytes,const void *b,size_t b_bytes) {
  size_t x=(size_t)a,y=(size_t)b;
  return x<=y ? y-x>=a_bytes : x-y>=b_bytes;
}

int cpu_gated_delta_scan_part(float *out, const float *q, const float *k,
    const float *v, const float *beta, const float *alpha, float *state,
    int start, int rows, int batch, int heads, int capacity, int key_dim,
    int value_dim, int alpha_dim, float *history, int group, int groups) {
  if(groups<1||group<0||group>=groups||!out||!q||!k||!v||!beta||!alpha||!state||start<0||rows<1||rows>capacity||capacity>256||
     batch<1||heads<1||key_dim<1||key_dim>256||value_dim<1||value_dim>256||
     (alpha_dim!=1&&alpha_dim!=key_dim))return 1;
  const size_t stride=(size_t)value_dim*key_dim,bh_count=(size_t)batch*heads;
  const size_t q_bytes=bh_count*capacity*key_dim*sizeof(float),v_bytes=bh_count*capacity*value_dim*sizeof(float);
  const size_t b_bytes=bh_count*capacity*sizeof(float),a_bytes=b_bytes*alpha_dim,s_bytes=bh_count*stride*sizeof(float);
  /* The general path snapshots state. Preserve its behavior for aliased inputs. */
  const int direct=ONPU_GDN_DIRECT_DECODE && rows==1 && capacity==1 && start>0 && !history && alpha_dim==1 &&
    key_dim%4==0 && value_dim%4==0 && scan_disjoint(state,s_bytes,q,q_bytes) && scan_disjoint(state,s_bytes,k,q_bytes) &&
    scan_disjoint(state,s_bytes,v,v_bytes) && scan_disjoint(state,s_bytes,beta,b_bytes) && scan_disjoint(state,s_bytes,alpha,a_bytes) &&
    scan_disjoint(state,s_bytes,out,v_bytes) && scan_disjoint(out,v_bytes,q,q_bytes) && scan_disjoint(out,v_bytes,k,q_bytes) &&
    scan_disjoint(out,v_bytes,v,v_bytes) && scan_disjoint(out,v_bytes,beta,b_bytes) && scan_disjoint(out,v_bytes,alpha,a_bytes);
  float columns[256*256] __attribute__((aligned(16)));
  for(size_t bh=(size_t)group;bh<bh_count;bh+=(size_t)groups) {
    float *s=state+bh*stride;
    if(direct) {
      size_t step=bh*capacity;
      direct_row(out+step*value_dim,q+step*key_dim,k+step*key_dim,v+step*value_dim,beta[step],alpha[step],s,key_dim,value_dim);
      continue;
    }
    for(int c=0;c<key_dim;c++)for(int r=0;r<value_dim;r++)
      columns[(size_t)c*value_dim+r]=start?s[(size_t)r*key_dim+c]:0;
    for(int t=0;t<rows;t++) {
      const size_t step=bh*capacity+t;
      const float *qt=q+step*key_dim,*kt=k+step*key_dim,*at=alpha+step*alpha_dim;
      const F4 b=(F4){beta[step],beta[step],beta[step],beta[step]};
      /* Share key/query loads across four vectors without changing each key reduction. */
      if(value_dim%64==0) {
        for(int r=0;r<value_dim;r+=64) {
          F4 predicted0=(F4){0,0,0,0};
          F4 predicted1=(F4){0,0,0,0};
          F4 predicted2=(F4){0,0,0,0};
          F4 predicted3=(F4){0,0,0,0};
          F4 predicted4=(F4){0,0,0,0};
          F4 predicted5=(F4){0,0,0,0};
          F4 predicted6=(F4){0,0,0,0};
          F4 predicted7=(F4){0,0,0,0};
          F4 predicted8=(F4){0,0,0,0};
          F4 predicted9=(F4){0,0,0,0};
          F4 predicted10=(F4){0,0,0,0};
          F4 predicted11=(F4){0,0,0,0};
          F4 predicted12=(F4){0,0,0,0};
          F4 predicted13=(F4){0,0,0,0};
          F4 predicted14=(F4){0,0,0,0};
          F4 predicted15=(F4){0,0,0,0};
          for(int c=0;c<key_dim;c++) {
            const float *p=columns+(size_t)c*value_dim+r;
            F4 key=(F4){kt[c],kt[c],kt[c],kt[c]};
            F4 old0=*(const F4*)(p+0);
            F4 old1=*(const F4*)(p+4);
            F4 old2=*(const F4*)(p+8);
            F4 old3=*(const F4*)(p+12);
            F4 old4=*(const F4*)(p+16);
            F4 old5=*(const F4*)(p+20);
            F4 old6=*(const F4*)(p+24);
            F4 old7=*(const F4*)(p+28);
            F4 old8=*(const F4*)(p+32);
            F4 old9=*(const F4*)(p+36);
            F4 old10=*(const F4*)(p+40);
            F4 old11=*(const F4*)(p+44);
            F4 old12=*(const F4*)(p+48);
            F4 old13=*(const F4*)(p+52);
            F4 old14=*(const F4*)(p+56);
            F4 old15=*(const F4*)(p+60);
            if(alpha_dim!=1) {
              F4 a=(F4){at[c],at[c],at[c],at[c]};
              old0=old0*a;
              old1=old1*a;
              old2=old2*a;
              old3=old3*a;
              old4=old4*a;
              old5=old5*a;
              old6=old6*a;
              old7=old7*a;
              old8=old8*a;
              old9=old9*a;
              old10=old10*a;
              old11=old11*a;
              old12=old12*a;
              old13=old13*a;
              old14=old14*a;
              old15=old15*a;
            }
            predicted0=predicted0+old0*key;
            predicted1=predicted1+old1*key;
            predicted2=predicted2+old2*key;
            predicted3=predicted3+old3*key;
            predicted4=predicted4+old4*key;
            predicted5=predicted5+old5*key;
            predicted6=predicted6+old6*key;
            predicted7=predicted7+old7*key;
            predicted8=predicted8+old8*key;
            predicted9=predicted9+old9*key;
            predicted10=predicted10+old10*key;
            predicted11=predicted11+old11*key;
            predicted12=predicted12+old12*key;
            predicted13=predicted13+old13*key;
            predicted14=predicted14+old14*key;
            predicted15=predicted15+old15*key;
          }
          F4 decay=(F4){at[0],at[0],at[0],at[0]};
          F4 value0=*(const F4*)(v+step*value_dim+r+0),delta0;
          F4 value1=*(const F4*)(v+step*value_dim+r+4),delta1;
          F4 value2=*(const F4*)(v+step*value_dim+r+8),delta2;
          F4 value3=*(const F4*)(v+step*value_dim+r+12),delta3;
          F4 value4=*(const F4*)(v+step*value_dim+r+16),delta4;
          F4 value5=*(const F4*)(v+step*value_dim+r+20),delta5;
          F4 value6=*(const F4*)(v+step*value_dim+r+24),delta6;
          F4 value7=*(const F4*)(v+step*value_dim+r+28),delta7;
          F4 value8=*(const F4*)(v+step*value_dim+r+32),delta8;
          F4 value9=*(const F4*)(v+step*value_dim+r+36),delta9;
          F4 value10=*(const F4*)(v+step*value_dim+r+40),delta10;
          F4 value11=*(const F4*)(v+step*value_dim+r+44),delta11;
          F4 value12=*(const F4*)(v+step*value_dim+r+48),delta12;
          F4 value13=*(const F4*)(v+step*value_dim+r+52),delta13;
          F4 value14=*(const F4*)(v+step*value_dim+r+56),delta14;
          F4 value15=*(const F4*)(v+step*value_dim+r+60),delta15;
          if(alpha_dim==1) {
            delta0=(value0-predicted0*decay)*b;
            delta1=(value1-predicted1*decay)*b;
            delta2=(value2-predicted2*decay)*b;
            delta3=(value3-predicted3*decay)*b;
            delta4=(value4-predicted4*decay)*b;
            delta5=(value5-predicted5*decay)*b;
            delta6=(value6-predicted6*decay)*b;
            delta7=(value7-predicted7*decay)*b;
            delta8=(value8-predicted8*decay)*b;
            delta9=(value9-predicted9*decay)*b;
            delta10=(value10-predicted10*decay)*b;
            delta11=(value11-predicted11*decay)*b;
            delta12=(value12-predicted12*decay)*b;
            delta13=(value13-predicted13*decay)*b;
            delta14=(value14-predicted14*decay)*b;
            delta15=(value15-predicted15*decay)*b;
          } else {
            delta0=(value0-predicted0)*b;
            delta1=(value1-predicted1)*b;
            delta2=(value2-predicted2)*b;
            delta3=(value3-predicted3)*b;
            delta4=(value4-predicted4)*b;
            delta5=(value5-predicted5)*b;
            delta6=(value6-predicted6)*b;
            delta7=(value7-predicted7)*b;
            delta8=(value8-predicted8)*b;
            delta9=(value9-predicted9)*b;
            delta10=(value10-predicted10)*b;
            delta11=(value11-predicted11)*b;
            delta12=(value12-predicted12)*b;
            delta13=(value13-predicted13)*b;
            delta14=(value14-predicted14)*b;
            delta15=(value15-predicted15)*b;
          }
          F4 result0=(F4){0,0,0,0};
          F4 result1=(F4){0,0,0,0};
          F4 result2=(F4){0,0,0,0};
          F4 result3=(F4){0,0,0,0};
          F4 result4=(F4){0,0,0,0};
          F4 result5=(F4){0,0,0,0};
          F4 result6=(F4){0,0,0,0};
          F4 result7=(F4){0,0,0,0};
          F4 result8=(F4){0,0,0,0};
          F4 result9=(F4){0,0,0,0};
          F4 result10=(F4){0,0,0,0};
          F4 result11=(F4){0,0,0,0};
          F4 result12=(F4){0,0,0,0};
          F4 result13=(F4){0,0,0,0};
          F4 result14=(F4){0,0,0,0};
          F4 result15=(F4){0,0,0,0};
          for(int c=0;c<key_dim;c++) {
            float *p=columns+(size_t)c*value_dim+r;
            float av=at[alpha_dim==1?0:c];
            F4 a=(F4){av,av,av,av};
            F4 key=(F4){kt[c],kt[c],kt[c],kt[c]},query=(F4){qt[c],qt[c],qt[c],qt[c]};
            F4 old0=*(const F4*)(p+0);
            F4 updated0=old0*a+delta0*key;
            *(F4*)(p+0)=updated0;
            result0=result0+updated0*query;
            F4 old1=*(const F4*)(p+4);
            F4 updated1=old1*a+delta1*key;
            *(F4*)(p+4)=updated1;
            result1=result1+updated1*query;
            F4 old2=*(const F4*)(p+8);
            F4 updated2=old2*a+delta2*key;
            *(F4*)(p+8)=updated2;
            result2=result2+updated2*query;
            F4 old3=*(const F4*)(p+12);
            F4 updated3=old3*a+delta3*key;
            *(F4*)(p+12)=updated3;
            result3=result3+updated3*query;
            F4 old4=*(const F4*)(p+16);
            F4 updated4=old4*a+delta4*key;
            *(F4*)(p+16)=updated4;
            result4=result4+updated4*query;
            F4 old5=*(const F4*)(p+20);
            F4 updated5=old5*a+delta5*key;
            *(F4*)(p+20)=updated5;
            result5=result5+updated5*query;
            F4 old6=*(const F4*)(p+24);
            F4 updated6=old6*a+delta6*key;
            *(F4*)(p+24)=updated6;
            result6=result6+updated6*query;
            F4 old7=*(const F4*)(p+28);
            F4 updated7=old7*a+delta7*key;
            *(F4*)(p+28)=updated7;
            result7=result7+updated7*query;
            F4 old8=*(const F4*)(p+32);
            F4 updated8=old8*a+delta8*key;
            *(F4*)(p+32)=updated8;
            result8=result8+updated8*query;
            F4 old9=*(const F4*)(p+36);
            F4 updated9=old9*a+delta9*key;
            *(F4*)(p+36)=updated9;
            result9=result9+updated9*query;
            F4 old10=*(const F4*)(p+40);
            F4 updated10=old10*a+delta10*key;
            *(F4*)(p+40)=updated10;
            result10=result10+updated10*query;
            F4 old11=*(const F4*)(p+44);
            F4 updated11=old11*a+delta11*key;
            *(F4*)(p+44)=updated11;
            result11=result11+updated11*query;
            F4 old12=*(const F4*)(p+48);
            F4 updated12=old12*a+delta12*key;
            *(F4*)(p+48)=updated12;
            result12=result12+updated12*query;
            F4 old13=*(const F4*)(p+52);
            F4 updated13=old13*a+delta13*key;
            *(F4*)(p+52)=updated13;
            result13=result13+updated13*query;
            F4 old14=*(const F4*)(p+56);
            F4 updated14=old14*a+delta14*key;
            *(F4*)(p+56)=updated14;
            result14=result14+updated14*query;
            F4 old15=*(const F4*)(p+60);
            F4 updated15=old15*a+delta15*key;
            *(F4*)(p+60)=updated15;
            result15=result15+updated15*query;
          }
          *(F4*)(out+step*value_dim+r+0)=result0;
          *(F4*)(out+step*value_dim+r+4)=result1;
          *(F4*)(out+step*value_dim+r+8)=result2;
          *(F4*)(out+step*value_dim+r+12)=result3;
          *(F4*)(out+step*value_dim+r+16)=result4;
          *(F4*)(out+step*value_dim+r+20)=result5;
          *(F4*)(out+step*value_dim+r+24)=result6;
          *(F4*)(out+step*value_dim+r+28)=result7;
          *(F4*)(out+step*value_dim+r+32)=result8;
          *(F4*)(out+step*value_dim+r+36)=result9;
          *(F4*)(out+step*value_dim+r+40)=result10;
          *(F4*)(out+step*value_dim+r+44)=result11;
          *(F4*)(out+step*value_dim+r+48)=result12;
          *(F4*)(out+step*value_dim+r+52)=result13;
          *(F4*)(out+step*value_dim+r+56)=result14;
          *(F4*)(out+step*value_dim+r+60)=result15;
        }
      } else if(value_dim%16==0) {
        for(int r=0;r<value_dim;r+=16) {
          F4 predicted0=(F4){0,0,0,0};
          F4 predicted1=(F4){0,0,0,0};
          F4 predicted2=(F4){0,0,0,0};
          F4 predicted3=(F4){0,0,0,0};
          for(int c=0;c<key_dim;c++) {
            const float *p=columns+(size_t)c*value_dim+r;
            F4 key=(F4){kt[c],kt[c],kt[c],kt[c]};
            F4 old0=*(const F4*)(p+0);
            F4 old1=*(const F4*)(p+4);
            F4 old2=*(const F4*)(p+8);
            F4 old3=*(const F4*)(p+12);
            if(alpha_dim!=1) {
              F4 a=(F4){at[c],at[c],at[c],at[c]};
              old0=old0*a;
              old1=old1*a;
              old2=old2*a;
              old3=old3*a;
            }
            predicted0=predicted0+old0*key;
            predicted1=predicted1+old1*key;
            predicted2=predicted2+old2*key;
            predicted3=predicted3+old3*key;
          }
          F4 decay=(F4){at[0],at[0],at[0],at[0]};
          F4 value0=*(const F4*)(v+step*value_dim+r+0),delta0;
          F4 value1=*(const F4*)(v+step*value_dim+r+4),delta1;
          F4 value2=*(const F4*)(v+step*value_dim+r+8),delta2;
          F4 value3=*(const F4*)(v+step*value_dim+r+12),delta3;
          if(alpha_dim==1) {
            delta0=(value0-predicted0*decay)*b;
            delta1=(value1-predicted1*decay)*b;
            delta2=(value2-predicted2*decay)*b;
            delta3=(value3-predicted3*decay)*b;
          } else {
            delta0=(value0-predicted0)*b;
            delta1=(value1-predicted1)*b;
            delta2=(value2-predicted2)*b;
            delta3=(value3-predicted3)*b;
          }
          F4 result0=(F4){0,0,0,0};
          F4 result1=(F4){0,0,0,0};
          F4 result2=(F4){0,0,0,0};
          F4 result3=(F4){0,0,0,0};
          for(int c=0;c<key_dim;c++) {
            float *p=columns+(size_t)c*value_dim+r;
            float decay_value=at[alpha_dim==1?0:c];
            F4 a=(F4){decay_value,decay_value,decay_value,decay_value};
            F4 key=(F4){kt[c],kt[c],kt[c],kt[c]},query=(F4){qt[c],qt[c],qt[c],qt[c]};
            F4 old0=*(const F4*)(p+0);
            F4 updated0=old0*a+delta0*key;
            *(F4*)(p+0)=updated0;
            result0=result0+updated0*query;
            F4 old1=*(const F4*)(p+4);
            F4 updated1=old1*a+delta1*key;
            *(F4*)(p+4)=updated1;
            result1=result1+updated1*query;
            F4 old2=*(const F4*)(p+8);
            F4 updated2=old2*a+delta2*key;
            *(F4*)(p+8)=updated2;
            result2=result2+updated2*query;
            F4 old3=*(const F4*)(p+12);
            F4 updated3=old3*a+delta3*key;
            *(F4*)(p+12)=updated3;
            result3=result3+updated3*query;
          }
          *(F4*)(out+step*value_dim+r+0)=result0;
          *(F4*)(out+step*value_dim+r+4)=result1;
          *(F4*)(out+step*value_dim+r+8)=result2;
          *(F4*)(out+step*value_dim+r+12)=result3;
        }
      } else {
        for(int r=0;r<value_dim;r+=4) {
          int count=value_dim-r<4?value_dim-r:4;
          F4 predicted=(F4){0,0,0,0};
          for(int c=0;c<key_dim;c++) {
            F4 old=load_rows(columns+(size_t)c*value_dim+r,count);
            F4 key=(F4){kt[c],kt[c],kt[c],kt[c]};
            if(alpha_dim!=1) {
              F4 a=(F4){at[c],at[c],at[c],at[c]};
              old=old*a;
            }
            predicted=predicted+old*key;
          }
          F4 value=load_rows(v+step*value_dim+r,count),delta;
          if(alpha_dim==1) {
            F4 a=(F4){at[0],at[0],at[0],at[0]};
            delta=(value-predicted*a)*b;
          } else delta=(value-predicted)*b;
          F4 result=(F4){0,0,0,0};
          for(int c=0;c<key_dim;c++) {
            float *p=columns+(size_t)c*value_dim+r;
            F4 old=load_rows(p,count);
            float decay=at[alpha_dim==1?0:c];
            F4 a=(F4){decay,decay,decay,decay},key=(F4){kt[c],kt[c],kt[c],kt[c]};
            F4 updated=old*a+delta*key;
            store_rows(p,updated,count);
            F4 query=(F4){qt[c],qt[c],qt[c],qt[c]};
            result=result+updated*query;
          }
          store_rows(out+step*value_dim+r,result,count);
        }
      }
      if(history)for(int r=0;r<value_dim;r++)for(int c=0;c<key_dim;c++)
        history[((size_t)t*bh_count+bh)*stride+(size_t)r*key_dim+c]=columns[(size_t)c*value_dim+r];
    }
    for(int r=0;r<value_dim;r++)for(int c=0;c<key_dim;c++)s[(size_t)r*key_dim+c]=columns[(size_t)c*value_dim+r];
    memset(out+(bh*capacity+rows)*value_dim,0,(size_t)(capacity-rows)*value_dim*sizeof(float));
  }
  return 0;
}

/* Independent-head CPU worker dispatch; each head keeps its serial recurrence. */
typedef int (*scan_parallel_fn)(void (*)(void **,int,int),void **,int);
typedef struct {
  float *out,*state,*history;
  const float *q,*k,*v,*beta,*alpha;
  int start,rows,batch,heads,capacity,key_dim,value_dim,alpha_dim;
  int done[4],errors[4];
} ScanWork;

static void scan_pool_part(void **args,int unused,int group) {
  (void)unused;
  ScanWork *w=(ScanWork*)args[0];
  if(group<0||group>=4)return;
  w->errors[group]=cpu_gated_delta_scan_part(w->out,w->q,w->k,w->v,w->beta,w->alpha,w->state,
    w->start,w->rows,w->batch,w->heads,w->capacity,w->key_dim,w->value_dim,w->alpha_dim,w->history,group,4);
  w->done[group]=1;
}

int cpu_gated_delta_scan(float *out,const float *q,const float *k,const float *v,
    const float *beta,const float *alpha,float *state,int start,int rows,int batch,int heads,
    int capacity,int key_dim,int value_dim,int alpha_dim,float *history) {
  return cpu_gated_delta_scan_part(out,q,k,v,beta,alpha,state,start,rows,batch,heads,capacity,
    key_dim,value_dim,alpha_dim,history,0,1);
}

/* dispatch_status: 0 serial, 1 pooled, 2 serial after pre-dispatch error,
   3 partial-dispatch error, 4 completed with a dispatcher error. */
int cpu_gated_delta_scan_pooled(float *out,const float *q,const float *k,const float *v,
    const float *beta,const float *alpha,float *state,int start,int rows,int batch,int heads,
    int capacity,int key_dim,int value_dim,int alpha_dim,float *history,
    scan_parallel_fn parallel,int *dispatch_status) {
  if(dispatch_status)*dispatch_status=0;
  if(!parallel||batch<1||heads<1||(size_t)batch*heads<4)
    return cpu_gated_delta_scan(out,q,k,v,beta,alpha,state,start,rows,batch,heads,capacity,key_dim,value_dim,alpha_dim,history);
  ScanWork work={.out=out,.state=state,.history=history,.q=q,.k=k,.v=v,.beta=beta,.alpha=alpha,
    .start=start,.rows=rows,.batch=batch,.heads=heads,.capacity=capacity,.key_dim=key_dim,
    .value_dim=value_dim,.alpha_dim=alpha_dim,.done={0,0,0,0},.errors={0,0,0,0}};
  /* The pool copies exactly four pointers for its workers. The work record
     remains valid until the synchronous pool call has joined every callback. */
  void *args[4]={&work,0,0,0};
  int error=parallel(scan_pool_part,args,0),completed=0;
  for(int group=0;group<4;group++)completed+=work.done[group]!=0;
  if(completed==4) {
    if(dispatch_status)*dispatch_status=error?4:1;
    if(error)return error;
    for(int group=0;group<4;group++)if(work.errors[group])return work.errors[group];
    return 0;
  }
  if(completed==0&&error) {
    if(dispatch_status)*dispatch_status=2;
    return cpu_gated_delta_scan(out,q,k,v,beta,alpha,state,start,rows,batch,heads,capacity,key_dim,value_dim,alpha_dim,history);
  }
  /* Recurrent state may already have changed: never replay serially here. */
  if(dispatch_status)*dispatch_status=3;
  return error?error:5;
}
