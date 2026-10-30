typedef unsigned long uintptr_t;
typedef unsigned long size_t;
extern void *memset(void*,int,size_t);

#ifndef HEADS
#define HEADS 16
#endif
#ifndef KV_HEADS
#define KV_HEADS 8
#endif
#ifndef DIM
#define DIM 128
#endif
#ifndef CONTEXT
#define CONTEXT 512
#endif
#ifndef SHARE_KV
#define SHARE_KV 0
#endif
#ifndef ROW_MAJOR
#define ROW_MAJOR 0
#endif
#ifndef TOKEN_MAJOR_OUTPUT
#define TOKEN_MAJOR_OUTPUT 0
#endif
#ifndef ATTENTION_POOL_ADDRESS
#define ATTENTION_POOL_ADDRESS 0
#endif
#ifndef ATTENTION_HELPER_API
#define ATTENTION_HELPER_API static
#endif

/* Bounded layouts: Qwen3 (16/8/128) and TinyLLaMA (32/4/64).
   rockchip_softmax_vector is prepended from rockchip_cpu.py by the caller. */
#define QUERY_GROUP (HEADS/KV_HEADS)

typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
typedef unsigned short ushort4 __attribute__((aligned(8),ext_vector_type(4)));
typedef void (*parallel_callback)(void **,int,int);
typedef int (*parallel_dispatch)(parallel_callback,void **,int);
static parallel_dispatch attention_parallel=(parallel_dispatch)(uintptr_t)ATTENTION_POOL_ADDRESS;
ATTENTION_HELPER_API void set_pool(unsigned long long address) { attention_parallel=(parallel_dispatch)(uintptr_t)address; }

static int bad4(float4 v) {
  return __builtin_reduce_or((v!=(float4)v)|(v>(float4)65504.0f)|(v<(float4)-65504.0f));
}

static int bad_half4(half4 v) {
  ushort4 bits=__builtin_bit_cast(ushort4,v);
  return __builtin_reduce_or((bits&(ushort4)0x7c00)==(ushort4)0x7c00)!=0;
}

static int geometry_ok(int capacity,int active,int total,int padded) {
  return capacity>=1 && capacity<=128 && active>=1 && active<=capacity && total>=active && total<=CONTEXT &&
    padded>=total && padded<=CONTEXT && (padded==64 || padded==128 || padded==256 || padded==512);
}

static int pack_part(int capacity,int active,int total,int padded,const float *q,const __fp16 *cache,
                     __fp16 *qa,__fp16 *ka,__fp16 *va,int head_first,int head_end) {
  for(int h=head_first;h<head_end;h++) {
    int g=SHARE_KV ? (h%QUERY_GROUP)*KV_HEADS+h/QUERY_GROUP : h;
    int w=SHARE_KV ? h/QUERY_GROUP : h;
    if(active<capacity)memset(qa+(size_t)g*capacity*DIM,0,(size_t)capacity*DIM*sizeof(__fp16));
    for(int r=0;r<active;r++) {
      int tile=r/64,local=r%64,m=capacity-tile*64;if(m>64)m=64;
      for(int c=0;c<DIM;c+=4) {
        float4 v=*(const float4*)(q+((size_t)h*capacity+r)*DIM+c);
        if(bad4(v))return 1;
#if ROW_MAJOR
        *(half4*)(qa+((size_t)g*capacity+r)*DIM+c)=__builtin_convertvector(v,half4);
#else
        *(half4*)(qa+(size_t)g*capacity*DIM+tile*64*DIM+(c/8)*m*8+local*8+c%8)=__builtin_convertvector(v,half4);
#endif
      }
    }
    if(SHARE_KV && h%QUERY_GROUP)continue;
    for(int t=0;t<padded;t++)for(int c=0;c<DIM;c+=4) {
      half4 v=t<total?*(const half4*)(cache+((size_t)(h/QUERY_GROUP)*CONTEXT+t)*DIM+c):(half4)0;
      if(bad_half4(v))return 2;
      *(half4*)(ka+(size_t)w*padded*DIM+(t/16)*DIM*16+(c/32)*512+(t%16)*32+c%32)=v;
    }
    for(int t=0;t<padded;t+=4)for(int c=0;c<DIM;c+=4) {
      const __fp16 *base=cache+(size_t)KV_HEADS*CONTEXT*DIM+((size_t)(h/QUERY_GROUP)*CONTEXT+t)*DIM+c;
      half4 a=t<total?*(const half4*)base:(half4)0;
      half4 b=t+1<total?*(const half4*)(base+DIM):(half4)0;
      half4 d=t+2<total?*(const half4*)(base+2*DIM):(half4)0;
      half4 e=t+3<total?*(const half4*)(base+3*DIM):(half4)0;
      if(bad_half4(a)|bad_half4(b)|bad_half4(d)|bad_half4(e))return 3;
      __fp16 *out=va+(size_t)w*DIM*padded+(c/16)*padded*16+(t/32)*512+(c%16)*32+t%32;
      *(half4*)out=(half4){a[0],b[0],d[0],e[0]};
      *(half4*)(out+32)=(half4){a[1],b[1],d[1],e[1]};
      *(half4*)(out+64)=(half4){a[2],b[2],d[2],e[2]};
      *(half4*)(out+96)=(half4){a[3],b[3],d[3],e[3]};
    }
  }
  return 0;
}

static int normalize_part(int capacity,int active,int total,int padded,const float *scores,__fp16 *prob,
                          int head_first,int head_end) {
  for(int h=head_first;h<head_end;h++) {
    int g=SHARE_KV ? (h%QUERY_GROUP)*KV_HEADS+h/QUERY_GROUP : h;
    memset(prob+(size_t)g*capacity*padded,0,(size_t)capacity*padded*sizeof(__fp16));
    for(int r=0;r<active;r++) {
      int tile=r/64,local=r%64,m=capacity-tile*64;if(m>64)m=64;
      int used=total-active+r+1;
      float p[512] __attribute__((aligned(16))),maximum=-__builtin_inff();
#if DIM==64
      const float score_scale=0x1p-3f;
#else
      const float score_scale=0x1.6a09e6p-4f;
#endif
      for(int t=0;t<used;t++) {
#if ROW_MAJOR
        float value=scores[((size_t)g*capacity+r)*padded+t]*score_scale;
#else
        float value=scores[(size_t)g*capacity*padded+tile*64*padded+(t/4)*m*4+local*4+t%4]*score_scale;
#endif
        if((__builtin_bit_cast(unsigned,value)&0x7f800000U)==0x7f800000U)return 11;
        p[t]=value;
        if(value>maximum)maximum=value;
      }
      rockchip_softmax_vector(p,used,maximum);
      int t=0;
      for(;t+3<used;t+=4) {
#if ROW_MAJOR
        *(half4*)(prob+((size_t)g*capacity+r)*padded+t)=__builtin_convertvector(*(const float4*)(p+t),half4);
#else
        *(half4*)(prob+(size_t)g*capacity*padded+tile*64*padded+(t/8)*m*8+local*8+t%8)=
          __builtin_convertvector(*(const float4*)(p+t),half4);
#endif
      }
      for(;t<used;t++) {
#if ROW_MAJOR
        prob[((size_t)g*capacity+r)*padded+t]=(__fp16)p[t];
#else
        prob[(size_t)g*capacity*padded+tile*64*padded+(t/8)*m*8+local*8+t%8]=(__fp16)p[t];
#endif
      }
    }
  }
  return 0;
}

static int unpack_result_part(int capacity,int active,int padded,const float *raw,float *out,
                              int head_first,int head_end) {
  int tile=64;if(65536/padded<tile)tile=65536/padded;
  for(int h=head_first;h<head_end;h++) {
    int g=SHARE_KV ? (h%QUERY_GROUP)*KV_HEADS+h/QUERY_GROUP : h;
#if !TOKEN_MAJOR_OUTPUT
    if(active<capacity)memset(out+(size_t)h*capacity*DIM,0,(size_t)capacity*DIM*sizeof(float));
#endif
    for(int r=0;r<active;r++) {
      int first=r/tile*tile,local=r-first,m=capacity-first;if(m>tile)m=tile;
      for(int c=0;c<DIM;c+=4) {
#if TOKEN_MAJOR_OUTPUT
        *(float4*)(out+((size_t)r*HEADS+h)*DIM+c)=
#else
        *(float4*)(out+((size_t)h*capacity+r)*DIM+c)=
#endif
#if ROW_MAJOR
          *(const float4*)(raw+((size_t)g*capacity+r)*DIM+c);
#else
          *(const float4*)(raw+(size_t)g*capacity*DIM+first*DIM+(c/4)*m*4+local*4);
#endif
      }
    }
  }
  return 0;
}

typedef struct {
  int phase,capacity,active,total,padded;
  const float *input;
  const __fp16 *cache;
  __fp16 *first,*second,*third;
  float *output;
  int errors[4];
} phase_args;

static void run_phase(void **args,int unused,int group) {
  (void)unused;
  phase_args *a=(phase_args*)args[0];
  int first=group*(HEADS/4),end=first+(HEADS/4),code=0;
  if(a->phase==0)code=pack_part(a->capacity,a->active,a->total,a->padded,a->input,a->cache,
                                a->first,a->second,a->third,first,end);
  else if(a->phase==1)code=normalize_part(a->capacity,a->active,a->total,a->padded,a->input,a->first,first,end);
  else code=unpack_result_part(a->capacity,a->active,a->padded,a->input,a->output,first,end);
  a->errors[group]=code;
}

static int dispatch_phase(phase_args *a) {
  if(!attention_parallel) {
    if(a->phase==0)return pack_part(a->capacity,a->active,a->total,a->padded,a->input,a->cache,
                                    a->first,a->second,a->third,0,HEADS);
    if(a->phase==1)return normalize_part(a->capacity,a->active,a->total,a->padded,a->input,a->first,0,HEADS);
    return unpack_result_part(a->capacity,a->active,a->padded,a->input,a->output,0,HEADS);
  }
  for(int i=0;i<4;i++)a->errors[i]=0;
  void *args[4]={a,0,0,0};
  int error=attention_parallel(run_phase,args,0);
  if(error)return 1000+error;
  for(int i=0;i<4;i++)if(a->errors[i])return a->errors[i];
  return 0;
}

ATTENTION_HELPER_API int pack(int capacity,int active,int total,int padded,const float *q,const __fp16 *cache,
         __fp16 *qa,__fp16 *ka,__fp16 *va) {
  if(!geometry_ok(capacity,active,total,padded))return 10;
  phase_args a={.phase=0,.capacity=capacity,.active=active,.total=total,.padded=padded,
    .input=q,.cache=cache,.first=qa,.second=ka,.third=va};
  return dispatch_phase(&a);
}

ATTENTION_HELPER_API int normalize(int capacity,int active,int total,int padded,const float *scores,__fp16 *prob) {
  if(!geometry_ok(capacity,active,total,padded))return 10;
  phase_args a={.phase=1,.capacity=capacity,.active=active,.total=total,.padded=padded,
    .input=scores,.first=prob};
  return dispatch_phase(&a);
}

ATTENTION_HELPER_API int unpack_result(int capacity,int active,int padded,const float *raw,float *out) {
  if(capacity<1 || capacity>128 || active<1 || active>capacity || padded<64 || padded>CONTEXT ||
     (padded!=64 && padded!=128 && padded!=256 && padded!=512))return 10;
#if TOKEN_MAJOR_OUTPUT
  if(active<capacity)memset(out+(size_t)active*HEADS*DIM,0,(size_t)(capacity-active)*HEADS*DIM*sizeof(float));
#endif
  phase_args a={.phase=2,.capacity=capacity,.active=active,.padded=padded,.input=raw,.output=out};
  return dispatch_phase(&a);
}
