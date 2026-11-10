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

/* Row-blocked re-tiling: one weight vector feeds eight rows and two adjacent
   column vectors. Each output still accumulates k ascending with the same
   `acc = acc + x*w` expression, so results are bit-identical to project_part. */
static void project_part_blocked(MLAProjection *w,int group,int groups) {
  for(int h=group;h<w->heads;h+=groups) {
    const __fp16 *wt=w->weight+(size_t)h*w->columns*w->depth;
    const float *xh=w->input+(size_t)h*w->capacity*w->depth;
    for(int row=0;row<w->rows;row+=8) {
      int count=w->rows-row<8?w->rows-row:8;
      const float *x=xh+(size_t)row*w->depth;
      for(int column=0;column<w->columns;column+=8) {
        MLA4 a00={0,0,0,0},a01={0,0,0,0},a10={0,0,0,0},a11={0,0,0,0};
        MLA4 a20={0,0,0,0},a21={0,0,0,0},a30={0,0,0,0},a31={0,0,0,0};
        MLA4 a40={0,0,0,0},a41={0,0,0,0},a50={0,0,0,0},a51={0,0,0,0};
        MLA4 a60={0,0,0,0},a61={0,0,0,0},a70={0,0,0,0},a71={0,0,0,0};
        const __fp16 *w0=wt+(size_t)column*w->depth,*w1=wt+(size_t)(column+4)*w->depth;
        for(int k=0;k<w->depth;k++) {
          MLA4 v0=__builtin_convertvector(*(const MLAH4*)(w0+(size_t)k*4),MLA4);
          MLA4 v1=__builtin_convertvector(*(const MLAH4*)(w1+(size_t)k*4),MLA4);
          float x0=x[k];MLA4 b0={x0,x0,x0,x0};
          a00=a00+b0*v0;a01=a01+b0*v1;
          if(count>1){float x1=x[w->depth+k];MLA4 b1={x1,x1,x1,x1};a10=a10+b1*v0;a11=a11+b1*v1;}
          if(count>2){float x2=x[2*w->depth+k];MLA4 b2={x2,x2,x2,x2};a20=a20+b2*v0;a21=a21+b2*v1;}
          if(count>3){float x3=x[3*w->depth+k];MLA4 b3={x3,x3,x3,x3};a30=a30+b3*v0;a31=a31+b3*v1;}
          if(count>4){float x4=x[4*w->depth+k];MLA4 b4={x4,x4,x4,x4};a40=a40+b4*v0;a41=a41+b4*v1;}
          if(count>5){float x5=x[5*w->depth+k];MLA4 b5={x5,x5,x5,x5};a50=a50+b5*v0;a51=a51+b5*v1;}
          if(count>6){float x6=x[6*w->depth+k];MLA4 b6={x6,x6,x6,x6};a60=a60+b6*v0;a61=a61+b6*v1;}
          if(count>7){float x7=x[7*w->depth+k];MLA4 b7={x7,x7,x7,x7};a70=a70+b7*v0;a71=a71+b7*v1;}
        }
        float *out=w->out+((size_t)h*w->capacity+row)*w->columns+column;
        *(MLA4*)out=a00;*(MLA4*)(out+4)=a01;
        if(count>1){float *o=out+w->columns;*(MLA4*)o=a10;*(MLA4*)(o+4)=a11;}
        if(count>2){float *o=out+2*w->columns;*(MLA4*)o=a20;*(MLA4*)(o+4)=a21;}
        if(count>3){float *o=out+3*w->columns;*(MLA4*)o=a30;*(MLA4*)(o+4)=a31;}
        if(count>4){float *o=out+4*w->columns;*(MLA4*)o=a40;*(MLA4*)(o+4)=a41;}
        if(count>5){float *o=out+5*w->columns;*(MLA4*)o=a50;*(MLA4*)(o+4)=a51;}
        if(count>6){float *o=out+6*w->columns;*(MLA4*)o=a60;*(MLA4*)(o+4)=a61;}
        if(count>7){float *o=out+7*w->columns;*(MLA4*)o=a70;*(MLA4*)(o+4)=a71;}
      }
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
static void project_worker_blocked(void **args,int unused,int group) {
  (void)unused;
  if(group<0||group>=4)__builtin_trap();
  project_part_blocked(args[0],group,4);
}
static int mla_project(float *out,const float *input,const __fp16 *weight,
                int heads,int capacity,int rows,int depth,int columns,uintptr_t parallel,int blocked) {
  if(heads<1||heads>128||rows<1||rows>capacity||capacity>128||
      depth<4||depth>4096||depth%4||columns<4||columns>4096||columns%4||blocked<0||blocked>1)return 1;
  /* The blocked tile writes two 4-column vectors at a time; other widths keep
     the shipped kernel (bit-identical either way, this only avoids an overrun). */
  if(blocked&&columns%8)blocked=0;
  /* The kernel writes all `rows` rows of every head; when rows==capacity no
     output element is left unwritten, so the zero fill can be skipped. */
  if(rows<capacity)for(size_t i=0;i<(size_t)heads*capacity*columns;i++)out[i]=0;
  MLAProjection work={out,input,weight,heads,capacity,rows,depth,columns};
#if MLA_FP16
  if(rows==1&&!parallel){project_part_fp16(&work,0,1);return 0;}
#endif
  if(!parallel){(blocked?project_part_blocked:project_part)(&work,0,1);return 0;}
  void *args[4]={&work,0,0,0};
  return ((MLAParallel)parallel)(blocked?project_worker_blocked:project_worker,args,0);
}
