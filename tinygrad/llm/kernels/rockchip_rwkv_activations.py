"""Exact vector decay for the reviewed RWKV7 CPU prefill program."""
import hashlib
from tinygrad.llm.kernels.rockchip_cpu import _vector_exp2_source,_vector_exp2_batch4_source
NAME='E_32_512_4_4'
def vector_decay_source(original:str,name:str,address:int) -> str:
 if (hashlib.sha256(original.encode()).hexdigest()!='d1defdd357786b4f9f6d99e94a876bbb5bdeac0f0bbb18957e78f9cb283b8528'
     or name!=NAME or type(address) is not int or not 0<address<2**64):raise ValueError('Unqualified RWKV decay kernel')
 original=original.replace('void '+NAME+'(','static void '+NAME+'_serial(',1)
 code=r"""
 typedef struct {float*out;const float*input;const __fp16*bias;int active;} SigmoidWork;
 static void sigmoid_part(void **args,int unused,int worker) {
  SigmoidWork*w=args[0];
  for(int row=worker*32;row<(worker+1)*32;row+=4)for(int c=0;c<2048;c+=4){
   float4 bias=__builtin_convertvector(*(const half4*)(w->bias+c),float4);
   float4 x0=row<w->active?*(const float4*)(w->input+row*2048+c):(float4)0;
   float4 x1=row+1<w->active?*(const float4*)(w->input+(row+1)*2048+c):(float4)0;
   float4 x2=row+2<w->active?*(const float4*)(w->input+(row+2)*2048+c):(float4)0;
   float4 x3=row+3<w->active?*(const float4*)(w->input+(row+3)*2048+c):(float4)0;
   float4 a0=(x0+bias)*-1.4426950216293335f,a1=(x1+bias)*-1.4426950216293335f;
   float4 a2=(x2+bias)*-1.4426950216293335f,a3=(x3+bias)*-1.4426950216293335f;
   float4 e0,e1,e2,e3;rockchip_exp2_batch4(a0,a1,a2,a3,&e0,&e1,&e2,&e3);
   float4 d0=-0.8750388026237488f/(1.0f+e0),d1=-0.8750388026237488f/(1.0f+e1);
   float4 d2=-0.8750388026237488f/(1.0f+e2),d3=-0.8750388026237488f/(1.0f+e3);
   rockchip_exp2_batch4(d0,d1,d2,d3,&e0,&e1,&e2,&e3);
   *(float4*)(w->out+row*2048+c)=row<w->active?e0:(float4)0;
   *(float4*)(w->out+(row+1)*2048+c)=row+1<w->active?e1:(float4)0;
   *(float4*)(w->out+(row+2)*2048+c)=row+2<w->active?e2:(float4)0;
   *(float4*)(w->out+(row+3)*2048+c)=row+3<w->active?e3:(float4)0;
  }
 }
 static int overlap(const void*a,unsigned long an,const void*b,unsigned long bn){
  unsigned long x=(unsigned long)a,y=(unsigned long)b;return x<=y?y-x<an:x-y<bn;
 }
 void E_32_512_4_4(float*out,float*input,__fp16*bias,const int active){
  if(overlap(out,1048576UL,input,1048576UL)||overlap(out,1048576UL,bias,4096UL)){
   E_32_512_4_4_serial(out,input,bias,active);return;
  }
  SigmoidWork work={out,input,bias,active};void*args[4]={&work,0,0,0};
  if(((int(*)(void(*)(void**,int,int),void**,int))ADDRESSULL)(sigmoid_part,args,0))
   E_32_512_4_4_serial(out,input,bias,active);
 }
"""
 return _vector_exp2_source()+_vector_exp2_batch4_source()+original+code.replace('ADDRESS',str(address))
