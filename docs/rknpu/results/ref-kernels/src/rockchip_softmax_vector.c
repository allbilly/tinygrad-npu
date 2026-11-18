typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  extern float expf(float);
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef int int4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));static __attribute__((always_inline)) float4 rockchip_exp2_slow(float4 x) {int4 nan=x!=x;
    float4 safe=((x==(float4)__builtin_inff())|(x==(float4)-__builtin_inff())|nan)?(float4)0:x;
    int4 q=__builtin_convertvector(safe+(safe<0?(float4)-.5f:(float4).5f),int4),q1=q>>1;
    float4 s=safe-__builtin_convertvector(q,float4);
    float4 y=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
      0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f)*__builtin_bit_cast(float4,(q1+127)<<23)*__builtin_bit_cast(float4,(q-q1+127)<<23);
    y=x>=128?(float4)__builtin_inff():y;
    y=x<-150?(float4)0:y;
    y=nan?(float4)__builtin_nanf(""):y;
    return y;}
  static __attribute__((always_inline)) float4 rockchip_exp2_vector(float4 x) {
    int4 magnitude=__builtin_bit_cast(int4,x)&(int4)0x7fffffff;
    if(__builtin_expect(__builtin_reduce_max(magnitude)>=0x42fa0000,0))return rockchip_exp2_slow(x);
    int4 q=__builtin_convertvector(x+(x<0?(float4)-.5f:(float4).5f),int4);
    float4 s=x-__builtin_convertvector(q,float4);
    float4 y=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
      0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f);
    return __builtin_bit_cast(float4,__builtin_bit_cast(int4,y)+q*(int4)0x800000);
  }static void rockchip_softmax_vector(float *values,int count,float maximum) {
    float sum=0;int t=0;
    for(;t+3<count;t+=4){float4 x;__builtin_memcpy(&x,values+t,16);
      float4 y=rockchip_exp2_vector((x-(float4)maximum)*(float4)1.4426950216293335f);
      __builtin_memcpy(values+t,&y,16);
      sum+=y[0];sum+=y[1];sum+=y[2];sum+=y[3];
    }
    for(;t<count;t++){values[t]=expf(values[t]-maximum);sum+=values[t];}
    for(t=0;t+3<count;t+=4){float4 x;__builtin_memcpy(&x,values+t,16);x=x/(float4)sum;__builtin_memcpy(values+t,&x,16);}
    for(;t<count;t++)values[t]/=sum;
  }
  
  extern float expf(float);
  static void attention_part(float*,const float*,const __fp16*,int,int,int,int);
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {float *out;const float *query;const __fp16 *cache;int total,rows,begin,end;} Work;
  static void *attention_worker(void*);static void pool_prefill_part(void **args,int total,int group) {
      attention_part(args[0],args[1],args[2],total,*(const int*)args[3],16*group/4,16*(group+1)/4);
    }
  void rockchip_cpu_attention(float *out,const float *query,const __fp16 *cache,const int *length,const int *rows) {
    const int total=*length,active_rows=*rows;
    if(active_rows<1 || active_rows>128 || total<active_rows || total>512) {
      for(int i=0;i<524288;i++)out[i]=__builtin_nanf("");return;
    }
    if(active_rows>=16){void *args[4]={out,(void*)query,(void*)cache,(void*)&active_rows};
      if(!((int(*)(void(*)(void**,int,int),void**,int))543132856320ULL)(pool_prefill_part,args,total))return;
    }if(active_rows<16) attention_part(out,query,cache,total,active_rows,0,16);
    else {
      Work work[4];thread_t threads[3];int created[3];
      for(int i=0;i<4;i++)work[i]=(Work){out,query,cache,total,active_rows,16*i/4,16*(i+1)/4};
      for(int i=0;i<3;i++){
        created[i]=pthread_create(&threads[i],(void*)0,attention_worker,&work[i])==0;
        if(!created[i])attention_part(out,query,cache,total,active_rows,work[i].begin,work[i].end);
      }
      attention_part(out,query,cache,total,active_rows,work[3].begin,work[3].end);
      for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);
    }
  }
  static void attention_part(float *out,const float *query,const __fp16 *cache,int total,int active_rows,int begin,int end) {
    
    for(int head=begin;head<end;head++) {
      const __fp16 *keys=cache+(head/4)*131072;
      const __fp16 *values=keys+524288;
      int row=0;
      for(;row+4<=active_rows;row+=4) {
    int used[4];for(int i=0;i<4;i++)used[i]=total-active_rows+row+i+1;
    float p[4][512],maximum[4]={-__builtin_inff(),-__builtin_inff(),-__builtin_inff(),-__builtin_inff()};
    const float4 *q0=(const float4*)(query+(head*128+row+0)*256);
const float4 *q1=(const float4*)(query+(head*128+row+1)*256);
const float4 *q2=(const float4*)(query+(head*128+row+2)*256);
const float4 *q3=(const float4*)(query+(head*128+row+3)*256);
    for(int t=0;t<used[3];t++) {
      const half4 *source=(const half4*)(keys+t*256);float4 a0={0},a1={0},a2={0},a3={0};
      for(int c=0;c<64;c++) {float4 key=__builtin_convertvector(source[c],float4);a0=a0+q0[c]*key;
a1=a1+q1[c]*key;
a2=a2+q2[c]*key;
a3=a3+q3[c]*key;}
      if(t<used[0]) {float v=(((a0[0]+a0[1])+a0[2])+a0[3])*0x1.0000000000000p-4f;
    
    p[0][t]=v;if(v>maximum[0])maximum[0]=v;}
if(t<used[1]) {float v=(((a1[0]+a1[1])+a1[2])+a1[3])*0x1.0000000000000p-4f;
    
    p[1][t]=v;if(v>maximum[1])maximum[1]=v;}
if(t<used[2]) {float v=(((a2[0]+a2[1])+a2[2])+a2[3])*0x1.0000000000000p-4f;
    
    p[2][t]=v;if(v>maximum[2])maximum[2]=v;}
if(t<used[3]) {float v=(((a3[0]+a3[1])+a3[2])+a3[3])*0x1.0000000000000p-4f;
    
    p[3][t]=v;if(v>maximum[3])maximum[3]=v;}
    }
    for(int i=0;i<4;i++)rockchip_softmax_vector(p[i],used[i],maximum[i]);
    for(int i=0;i<4;i++) {
      for(int t=used[i];t<used[3];t++)p[i][t]=0;
    }
    for(int c=0;c<64;c+=2) {
      float4 a00={0},a01={0},a10={0},a11={0},a20={0},a21={0},a30={0},a31={0};
      for(int t=0;t<used[0];t++) {const half4 *value=(const half4*)(values+t*256);float4 v0=__builtin_convertvector(value[c+0],float4);
float4 v1=__builtin_convertvector(value[c+1],float4);a00=a00+v0*(float4)p[0][t];
a01=a01+v1*(float4)p[0][t];
a10=a10+v0*(float4)p[1][t];
a11=a11+v1*(float4)p[1][t];
a20=a20+v0*(float4)p[2][t];
a21=a21+v1*(float4)p[2][t];
a30=a30+v0*(float4)p[3][t];
a31=a31+v1*(float4)p[3][t];}
      for(int t=used[0];t<used[3];t++) {const half4 *value=(const half4*)(values+t*256);float4 v0=__builtin_convertvector(value[c+0],float4);
float4 v1=__builtin_convertvector(value[c+1],float4);if(t<used[0]) {a00=a00+v0*(float4)p[0][t];a01=a01+v1*(float4)p[0][t];}
if(t<used[1]) {a10=a10+v0*(float4)p[1][t];a11=a11+v1*(float4)p[1][t];}
if(t<used[2]) {a20=a20+v0*(float4)p[2][t];a21=a21+v1*(float4)p[2][t];}
if(t<used[3]) {a30=a30+v0*(float4)p[3][t];a31=a31+v1*(float4)p[3][t];}}
      ((float4*)(out+((row+0)*16+head)*256))[c+0]=a00;
((float4*)(out+((row+0)*16+head)*256))[c+1]=a01;
((float4*)(out+((row+1)*16+head)*256))[c+0]=a10;
((float4*)(out+((row+1)*16+head)*256))[c+1]=a11;
((float4*)(out+((row+2)*16+head)*256))[c+0]=a20;
((float4*)(out+((row+2)*16+head)*256))[c+1]=a21;
((float4*)(out+((row+3)*16+head)*256))[c+0]=a30;
((float4*)(out+((row+3)*16+head)*256))[c+1]=a31;
    }
  }
      for(;row<active_rows;row++) {
      const int used=total-active_rows+row+1;
      const int first=0;
      const float4 *q=(const float4*)(query+(head*128+row)*256);
      float scores[512],maximum=-__builtin_inff(),denominator=0.0f;
      for(int t=first;t<used;t++) {
        const half4 *key=(const half4*)(keys+t*256);float4 acc={0};
        for(int c=0;c<64;c++)acc=acc+q[c]*__builtin_convertvector(key[c],float4);
        float score=((acc[0]+acc[1])+acc[2])+acc[3];
        scores[t]=score*0x1.0000000000000p-4f;
        
        if(scores[t]>maximum)maximum=scores[t];
      }
      rockchip_softmax_vector(scores+first,used-first,maximum);denominator=1.0f;
      float4 accum[64]={{0}};
      for(int t=first;t<used;t++) {
        const half4 *value=(const half4*)(values+t*256);float p=scores[t]/denominator;
        for(int c=0;c<64;c++)accum[c]=accum[c]+(float4){p,p,p,p}*__builtin_convertvector(value[c],float4);
      }
      for(int c=0;c<64;c++)((float4*)(out+(row*16+head)*256))[c]=accum[c];
      }
      for(;row<128;row++)for(int c=0;c<64;c++)((float4*)(out+(row*16+head)*256))[c]=(float4)0;
    }
  }
  static void *attention_worker(void *opaque) {
    Work *w=(Work*)opaque;attention_part(w->out,w->query,w->cache,w->total,w->rows,w->begin,w->end);return (void*)0;
  }