/* Bounded CPU MLA projections. FP32 reduction order is sequential K. */
#include <stddef.h>
#include <stdint.h>
typedef float MLA4 __attribute__((vector_size(16),aligned(16)));
typedef __fp16 MLAH4 __attribute__((vector_size(8),aligned(8)));
typedef int (*MLAParallel)(void (*)(void**,int,int),void**,int);
typedef struct {
  float *out;
  const float *input;
  const __fp16 *weight;
  int heads,capacity,rows,depth,columns;
} MLAProjection;

/* Weight layout: [head, output_channel/4, input_channel, output_channel%4].
   Four query rows share each converted weight vector. Each output retains
   sequential K accumulation under the caller's CPU floating-point contract. */
static void project_part(MLAProjection *w,int group,int groups) {
  for(int h=group;h<w->heads;h+=groups)for(int row=0;row<w->rows;row+=4) {
    int count=w->rows-row<4?w->rows-row:4;
    const float *x=w->input+((size_t)h*w->capacity+row)*w->depth;
    for(int column=0;column<w->columns;column+=4) {
      MLA4 a0={0,0,0,0},a1={0,0,0,0},a2={0,0,0,0},a3={0,0,0,0};
      const __fp16 *weights=w->weight+((size_t)h*w->columns+column)*w->depth;
      for(int k=0;k<w->depth;k++) {
        MLA4 weight=__builtin_convertvector(*(const MLAH4*)(weights+(size_t)k*4),MLA4);
        float value=x[k];a0=a0+(MLA4){value,value,value,value}*weight;
        if(count>1){value=x[w->depth+k];a1=a1+(MLA4){value,value,value,value}*weight;}
        if(count>2){value=x[2*w->depth+k];a2=a2+(MLA4){value,value,value,value}*weight;}
        if(count>3){value=x[3*w->depth+k];a3=a3+(MLA4){value,value,value,value}*weight;}
      }
      float *out=w->out+((size_t)h*w->capacity+row)*w->columns+column;
      *(MLA4*)out=a0;
      if(count>1)*(MLA4*)(out+w->columns)=a1;
      if(count>2)*(MLA4*)(out+2*w->columns)=a2;
      if(count>3)*(MLA4*)(out+3*w->columns)=a3;
    }
  }
}
#if MLA_FP16
/* Decode-only FP16 accumulation matching RKLLM's captured fmla v*.8h kv_b dot
   (count 160 = key 96 + value 64); prefill keeps the FP32 sequential-K path. */
static void project_part_fp16(MLAProjection *w,int group,int groups) {
  for(int h=group;h<w->heads;h+=groups) {
    const float *x=w->input+((size_t)h*w->capacity)*w->depth;
    for(int column=0;column<w->columns;column+=4) {
      MLAH4 a={0,0,0,0};
      const __fp16 *weights=w->weight+((size_t)h*w->columns+column)*w->depth;
      for(int k=0;k<w->depth;k++) {
        __fp16 value=(__fp16)x[k];
        a=a+*(const MLAH4*)(weights+(size_t)k*4)*(MLAH4){value,value,value,value};
      }
      *(MLA4*)(w->out+((size_t)h*w->capacity)*w->columns+column)=__builtin_convertvector(a,MLA4);
    }
  }
}
#endif

static void project_worker(void **args,int unused,int group) {
  (void)unused;
  if(group<0||group>=4)__builtin_trap();
  project_part(args[0],group,4);
}
static int mla_project(float *out,const float *input,const __fp16 *weight,
                int heads,int capacity,int rows,int depth,int columns,uintptr_t parallel) {
  if(heads<1||heads>128||rows<1||rows>capacity||capacity>128||
      depth<4||depth>4096||depth%4||columns<4||columns>4096||columns%4)return 1;
  for(size_t i=0;i<(size_t)heads*capacity*columns;i++)out[i]=0;
  MLAProjection work={out,input,weight,heads,capacity,rows,depth,columns};
#if MLA_FP16
  if(rows==1&&!parallel){project_part_fp16(&work,0,1);return 0;}
#endif
  if(!parallel){project_part(&work,0,1);return 0;}
  void *args[4]={&work,0,0,0};
  return ((MLAParallel)parallel)(project_worker,args,0);
}
