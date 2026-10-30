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

static int cpu_gated_delta_scan_part(float *out, const float *q, const float *k,
    const float *v, const float *beta, const float *alpha, float *state,
    int start, int rows, int batch, int heads, int capacity, int key_dim,
    int value_dim, int alpha_dim, float *history, int group, int groups) {
  if(groups<1||group<0||group>=groups||!out||!q||!k||!v||!beta||!alpha||!state||start<0||rows<1||rows>capacity||capacity>256||
     batch<1||heads<1||key_dim<1||key_dim>256||value_dim<1||value_dim>256||
     (alpha_dim!=1&&alpha_dim!=key_dim))return 1;
  const size_t stride=(size_t)value_dim*key_dim,bh_count=(size_t)batch*heads;
  float columns[256*256] __attribute__((aligned(16)));
  for(size_t bh=(size_t)group;bh<bh_count;bh+=(size_t)groups) {
    float *s=state+bh*stride;
    for(int c=0;c<key_dim;c++)for(int r=0;r<value_dim;r++)
      columns[(size_t)c*value_dim+r]=start?s[(size_t)r*key_dim+c]:0;
    for(int t=0;t<rows;t++) {
      const size_t step=bh*capacity+t;
      const float *qt=q+step*key_dim,*kt=k+step*key_dim,*at=alpha+step*alpha_dim;
      const F4 b=(F4){beta[step],beta[step],beta[step],beta[step]};
      /* Share key/query loads across four vectors without changing each key reduction. */
      if(value_dim%16==0) {
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
