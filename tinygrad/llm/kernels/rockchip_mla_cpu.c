/* Bounded CPU MLA projections. FP32 reduction order is sequential K. */
#include <stddef.h>
#include <stdint.h>
typedef float MLA4 __attribute__((vector_size(16),aligned(16)));
typedef __fp16 MLAH4 __attribute__((vector_size(8),aligned(8)));
typedef float MLA2 __attribute__((vector_size(8),aligned(8)));
typedef int (*MLAParallel)(void (*)(void**,int,int),void**,int);
typedef struct {
  float *out;
  const float *input;
  const void *weight; /* FP16 table for selectors 0-2, FP32 copy for selector 3 */
  int heads,capacity,rows,depth,columns;
  int transposed; /* value projection only: write rows-major (row,head,column) */
} MLAProjection;

/* Output element offset for one (head,row,column-group start).
   transposed==0 keeps the shipped head-major layout; transposed==1 writes the
   row-major layout the attention-output linear consumes, so no transpose copy
   is materialised downstream. Values and arithmetic are unchanged. */
static size_t mla_out_offset(const MLAProjection *w,int head,int row,int column) {
  return w->transposed ? ((size_t)row*w->heads+head)*w->columns+column
                       : ((size_t)head*w->capacity+row)*w->columns+column;
}

/* Stride between consecutive rows in the output layout (head-major: one row of
   the head's block; transposed: one row of the whole (rows, heads*columns) map). */
#define MLA_ROW_STRIDE(w) ((w)->transposed ? (size_t)(w)->heads*(w)->columns : (size_t)(w)->columns)

/* Weight layout: [head, output_channel/4, input_channel, output_channel%4].
   Four query rows share each converted weight vector. Each output retains
   sequential K accumulation under the caller's CPU floating-point contract. */
static void project_part(MLAProjection *w,int group,int groups) {
  for(int h=group;h<w->heads;h+=groups)for(int row=0;row<w->rows;row+=4) {
    int count=w->rows-row<4?w->rows-row:4;
    const float *x=w->input+((size_t)h*w->capacity+row)*w->depth;
    for(int column=0;column<w->columns;column+=4) {
      MLA4 a0={0,0,0,0},a1={0,0,0,0},a2={0,0,0,0},a3={0,0,0,0};
      const __fp16 *weights=(const __fp16*)w->weight+((size_t)h*w->columns+column)*w->depth;
      for(int k=0;k<w->depth;k++) {
        MLA4 weight=__builtin_convertvector(*(const MLAH4*)(weights+(size_t)k*4),MLA4);
        float value=x[k];a0=a0+(MLA4){value,value,value,value}*weight;
        if(count>1){value=x[w->depth+k];a1=a1+(MLA4){value,value,value,value}*weight;}
        if(count>2){value=x[2*w->depth+k];a2=a2+(MLA4){value,value,value,value}*weight;}
        if(count>3){value=x[3*w->depth+k];a3=a3+(MLA4){value,value,value,value}*weight;}
      }
      float *out=w->out+mla_out_offset(w,h,row,column);
      *(MLA4*)out=a0;
      if(count>1)*(MLA4*)(out+MLA_ROW_STRIDE(w))=a1;
      if(count>2)*(MLA4*)(out+2*MLA_ROW_STRIDE(w))=a2;
      if(count>3)*(MLA4*)(out+3*MLA_ROW_STRIDE(w))=a3;
    }
  }
}

/* Row-blocked re-tiling: one weight vector feeds eight rows and two adjacent
   column vectors. Each output still accumulates k ascending with the same
   `acc = acc + x*w` expression, so results are bit-identical to project_part. */
static void project_part_blocked(MLAProjection *w,int group,int groups) {
  for(int h=group;h<w->heads;h+=groups) {
    const __fp16 *wt=(const __fp16*)w->weight+(size_t)h*w->columns*w->depth;
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
        float *out=w->out+mla_out_offset(w,h,row,column);
        *(MLA4*)out=a00;*(MLA4*)(out+4)=a01;
        if(count>1){float *o=out+MLA_ROW_STRIDE(w);*(MLA4*)o=a10;*(MLA4*)(o+4)=a11;}
        if(count>2){float *o=out+2*MLA_ROW_STRIDE(w);*(MLA4*)o=a20;*(MLA4*)(o+4)=a21;}
        if(count>3){float *o=out+3*MLA_ROW_STRIDE(w);*(MLA4*)o=a30;*(MLA4*)(o+4)=a31;}
        if(count>4){float *o=out+4*MLA_ROW_STRIDE(w);*(MLA4*)o=a40;*(MLA4*)(o+4)=a41;}
        if(count>5){float *o=out+5*MLA_ROW_STRIDE(w);*(MLA4*)o=a50;*(MLA4*)(o+4)=a51;}
        if(count>6){float *o=out+6*MLA_ROW_STRIDE(w);*(MLA4*)o=a60;*(MLA4*)(o+4)=a61;}
        if(count>7){float *o=out+7*MLA_ROW_STRIDE(w);*(MLA4*)o=a70;*(MLA4*)(o+4)=a71;}
      }
    }
  }
}

/* Two-k-step variant of project_part_blocked (ONPU_MLA_PROJ_KPAIR). The 8-row x
   two-column-vector tile is unchanged; the inner loop consumes two consecutive k
   steps per iteration so one FP16 weight vector load feeds two accumulators and
   the two gate broadcasts come from a single 8-byte load. Every output element
   still runs `acc = acc + x[k]*w[k]` in ascending k with the same operand order:
   the pair only interleaves two INDEPENDENT accumulator updates, which the
   compiler cannot reassociate without fast-math. No value is computed from a
   different expression, so the results are bit-identical. */
static void project_part_kpair(MLAProjection *w,int group,int groups) {
  for(int h=group;h<w->heads;h+=groups) {
    const __fp16 *wt=(const __fp16*)w->weight+(size_t)h*w->columns*w->depth;
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
        for(int k=0;k<w->depth;k+=2) {
          MLA4 c0=__builtin_convertvector(*(const MLAH4*)(w0+(size_t)k*4),MLA4);
          MLA4 d0=__builtin_convertvector(*(const MLAH4*)(w0+(size_t)(k+1)*4),MLA4);
          MLA4 c1=__builtin_convertvector(*(const MLAH4*)(w1+(size_t)k*4),MLA4);
          MLA4 d1=__builtin_convertvector(*(const MLAH4*)(w1+(size_t)(k+1)*4),MLA4);
#define MLA_K2(R,A0,A1)                                                        \
          { MLA2 xs=*(const MLA2*)(x+(size_t)(R)*w->depth+k);                    \
            MLA4 b0={xs[0],xs[0],xs[0],xs[0]},b1={xs[1],xs[1],xs[1],xs[1]};      \
            A0=A0+b0*c0;A0=A0+b1*d0;A1=A1+b0*c1;A1=A1+b1*d1; }
          MLA_K2(0,a00,a01);
          if(count>1)MLA_K2(1,a10,a11);
          if(count>2)MLA_K2(2,a20,a21);
          if(count>3)MLA_K2(3,a30,a31);
          if(count>4)MLA_K2(4,a40,a41);
          if(count>5)MLA_K2(5,a50,a51);
          if(count>6)MLA_K2(6,a60,a61);
          if(count>7)MLA_K2(7,a70,a71);
#undef MLA_K2
        }
        float *out=w->out+mla_out_offset(w,h,row,column);
        *(MLA4*)out=a00;*(MLA4*)(out+4)=a01;
        if(count>1){float *o=out+MLA_ROW_STRIDE(w);*(MLA4*)o=a10;*(MLA4*)(o+4)=a11;}
        if(count>2){float *o=out+2*MLA_ROW_STRIDE(w);*(MLA4*)o=a20;*(MLA4*)(o+4)=a21;}
        if(count>3){float *o=out+3*MLA_ROW_STRIDE(w);*(MLA4*)o=a30;*(MLA4*)(o+4)=a31;}
        if(count>4){float *o=out+4*MLA_ROW_STRIDE(w);*(MLA4*)o=a40;*(MLA4*)(o+4)=a41;}
        if(count>5){float *o=out+5*MLA_ROW_STRIDE(w);*(MLA4*)o=a50;*(MLA4*)(o+4)=a51;}
        if(count>6){float *o=out+6*MLA_ROW_STRIDE(w);*(MLA4*)o=a60;*(MLA4*)(o+4)=a61;}
        if(count>7){float *o=out+7*MLA_ROW_STRIDE(w);*(MLA4*)o=a70;*(MLA4*)(o+4)=a71;}
      }
    }
  }
}
/* Identical to `project_part_kpair` except the weight table is FP32: the four
   per-2-k-step `fcvtl` conversions and their 8-byte loads become plain 16-byte
   loads. fp16->fp32 conversion is exact, so with the same table values every
   operand (and therefore every bit of the result) is unchanged. Requires
   columns%8==0 and depth%2==0 (checked by the dispatcher).

   The weight pointer arithmetic below is in FLOAT units: `w0` and `w1` step by
   `depth` floats (the table stride is heads*columns*depth floats), because the
   same (column, k) indices address the fp32 table the fp16 kernel indexes with
   the identical integer offsets. An fp16-unit offset would be half as wide and
   read the wrong head for head>=1. */
static void project_part_f32w(MLAProjection *w,int group,int groups) {
  for(int h=group;h<w->heads;h+=groups) {
    const float *wt=(const float*)w->weight+(size_t)h*w->columns*w->depth;
    const float *xh=w->input+(size_t)h*w->capacity*w->depth;
    for(int row=0;row<w->rows;row+=8) {
      int count=w->rows-row<8?w->rows-row:8;
      const float *x=xh+(size_t)row*w->depth;
      for(int column=0;column<w->columns;column+=8) {
        MLA4 a00={0,0,0,0},a01={0,0,0,0},a10={0,0,0,0},a11={0,0,0,0};
        MLA4 a20={0,0,0,0},a21={0,0,0,0},a30={0,0,0,0},a31={0,0,0,0};
        MLA4 a40={0,0,0,0},a41={0,0,0,0},a50={0,0,0,0},a51={0,0,0,0};
        MLA4 a60={0,0,0,0},a61={0,0,0,0},a70={0,0,0,0},a71={0,0,0,0};
        const float *w0=wt+(size_t)column*w->depth,*w1=wt+(size_t)(column+4)*w->depth;
        for(int k=0;k<w->depth;k+=2) {
          MLA4 c0=*(const MLA4*)(w0+(size_t)k*4);
          MLA4 d0=*(const MLA4*)(w0+(size_t)(k+1)*4);
          MLA4 c1=*(const MLA4*)(w1+(size_t)k*4);
          MLA4 d1=*(const MLA4*)(w1+(size_t)(k+1)*4);
#define MLA_K2(R,A0,A1)                                                        \
          { MLA2 xs=*(const MLA2*)(x+(size_t)(R)*w->depth+k);                    \
            MLA4 b0={xs[0],xs[0],xs[0],xs[0]},b1={xs[1],xs[1],xs[1],xs[1]};      \
            A0=A0+b0*c0;A0=A0+b1*d0;A1=A1+b0*c1;A1=A1+b1*d1; }
          MLA_K2(0,a00,a01);
          if(count>1)MLA_K2(1,a10,a11);
          if(count>2)MLA_K2(2,a20,a21);
          if(count>3)MLA_K2(3,a30,a31);
          if(count>4)MLA_K2(4,a40,a41);
          if(count>5)MLA_K2(5,a50,a51);
          if(count>6)MLA_K2(6,a60,a61);
          if(count>7)MLA_K2(7,a70,a71);
#undef MLA_K2
        }
        float *out=w->out+mla_out_offset(w,h,row,column);
        *(MLA4*)out=a00;*(MLA4*)(out+4)=a01;
        if(count>1){float *o=out+MLA_ROW_STRIDE(w);*(MLA4*)o=a10;*(MLA4*)(o+4)=a11;}
        if(count>2){float *o=out+2*MLA_ROW_STRIDE(w);*(MLA4*)o=a20;*(MLA4*)(o+4)=a21;}
        if(count>3){float *o=out+3*MLA_ROW_STRIDE(w);*(MLA4*)o=a30;*(MLA4*)(o+4)=a31;}
        if(count>4){float *o=out+4*MLA_ROW_STRIDE(w);*(MLA4*)o=a40;*(MLA4*)(o+4)=a41;}
        if(count>5){float *o=out+5*MLA_ROW_STRIDE(w);*(MLA4*)o=a50;*(MLA4*)(o+4)=a51;}
        if(count>6){float *o=out+6*MLA_ROW_STRIDE(w);*(MLA4*)o=a60;*(MLA4*)(o+4)=a61;}
        if(count>7){float *o=out+7*MLA_ROW_STRIDE(w);*(MLA4*)o=a70;*(MLA4*)(o+4)=a71;}
      }
    }
  }
}

/* Full-tile (count==8) specialization of `project_part_f32w`: the seven
   `if(count>N)` guards sit outside the k loop instead of inside it, and every
   row's two accumulators are updated unconditionally. Arithmetic, operand order
   and ascending-k accumulation are untouched, so the result is bit-identical to
   `project_part_f32w`; a partial height tile defers to the guarded variant. */
static void project_part_f32w8(MLAProjection *w,int group,int groups) {
  for(int h=group;h<w->heads;h+=groups) {
    const float *wt=(const float*)w->weight+(size_t)h*w->columns*w->depth;
    const float *xh=w->input+(size_t)h*w->capacity*w->depth;
    for(int row=0;row<w->rows;row+=8) {
      int count=w->rows-row<8?w->rows-row:8;
      if(count!=8){project_part_f32w(w,group,groups);return;}
      const float *x=xh+(size_t)row*w->depth;
      for(int column=0;column<w->columns;column+=8) {
        MLA4 a00={0,0,0,0},a01={0,0,0,0},a10={0,0,0,0},a11={0,0,0,0};
        MLA4 a20={0,0,0,0},a21={0,0,0,0},a30={0,0,0,0},a31={0,0,0,0};
        MLA4 a40={0,0,0,0},a41={0,0,0,0},a50={0,0,0,0},a51={0,0,0,0};
        MLA4 a60={0,0,0,0},a61={0,0,0,0},a70={0,0,0,0},a71={0,0,0,0};
        const float *w0=wt+(size_t)column*w->depth,*w1=wt+(size_t)(column+4)*w->depth;
        for(int k=0;k<w->depth;k+=2) {
          MLA4 c0=*(const MLA4*)(w0+(size_t)k*4);
          MLA4 d0=*(const MLA4*)(w0+(size_t)(k+1)*4);
          MLA4 c1=*(const MLA4*)(w1+(size_t)k*4);
          MLA4 d1=*(const MLA4*)(w1+(size_t)(k+1)*4);
#define MLA_K2(R,A0,A1)                                                        \
          { MLA2 xs=*(const MLA2*)(x+(size_t)(R)*w->depth+k);                    \
            MLA4 b0={xs[0],xs[0],xs[0],xs[0]},b1={xs[1],xs[1],xs[1],xs[1]};      \
            A0=A0+b0*c0;A0=A0+b1*d0;A1=A1+b0*c1;A1=A1+b1*d1; }
          MLA_K2(0,a00,a01); MLA_K2(1,a10,a11); MLA_K2(2,a20,a21); MLA_K2(3,a30,a31);
          MLA_K2(4,a40,a41); MLA_K2(5,a50,a51); MLA_K2(6,a60,a61); MLA_K2(7,a70,a71);
#undef MLA_K2
        }
        float *out=w->out+mla_out_offset(w,h,row,column);
        *(MLA4*)out=a00;*(MLA4*)(out+4)=a01;
        {float *o=out+MLA_ROW_STRIDE(w);*(MLA4*)o=a10;*(MLA4*)(o+4)=a11;}
        {float *o=out+2*MLA_ROW_STRIDE(w);*(MLA4*)o=a20;*(MLA4*)(o+4)=a21;}
        {float *o=out+3*MLA_ROW_STRIDE(w);*(MLA4*)o=a30;*(MLA4*)(o+4)=a31;}
        {float *o=out+4*MLA_ROW_STRIDE(w);*(MLA4*)o=a40;*(MLA4*)(o+4)=a41;}
        {float *o=out+5*MLA_ROW_STRIDE(w);*(MLA4*)o=a50;*(MLA4*)(o+4)=a51;}
        {float *o=out+6*MLA_ROW_STRIDE(w);*(MLA4*)o=a60;*(MLA4*)(o+4)=a61;}
        {float *o=out+7*MLA_ROW_STRIDE(w);*(MLA4*)o=a70;*(MLA4*)(o+4)=a71;}
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
      const __fp16 *weights=(const __fp16*)w->weight+((size_t)h*w->columns+column)*w->depth;
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
static void project_worker_kpair(void **args,int unused,int group) {
  (void)unused;
  if(group<0||group>=4)__builtin_trap();
  project_part_kpair(args[0],group,4);
}
static void project_worker_f32w(void **args,int unused,int group) {
  (void)unused;
  if(group<0||group>=4)__builtin_trap();
  /* Full 8-row tiles use the hoisted-guard specialization; a partial tile defers. */
  project_part_f32w8(args[0],group,4);
}
/* `tiling` is the shipped selector argument: 0 = shipped 4x4, 1 = blocked 8x8,
   2 = blocked 8x8 with the two-k-step inner loop, 3 = the same k-pair tile over
   an FP32 weight table. Keeping it one integer leaves the emitted call site text
   unchanged from the shipped wrapper.
   The weight table is `const void *` because selector 3 reads it as FP32 while every
   other selector reads it as FP16. Both spellings describe the same table values (an
   fp16 -> fp32 conversion is exact), and keeping one parameter type means the
   emitted wrapper needs no cast that could hide a wrong-dtype wiring bug; each
   kernel casts the table to its own element type. */
static int mla_project(float *out,const float *input,const void *weight,
                int heads,int capacity,int rows,int depth,int columns,uintptr_t parallel,int tiling,
                int transposed) {
  if(heads<1||heads>128||rows<1||rows>capacity||capacity>128||
      depth<4||depth>4096||depth%4||columns<4||columns>4096||columns%4||tiling<0||tiling>3||transposed<0||transposed>1)return 1;
  /* The blocked tile writes two 4-column vectors at a time; the k-pair tile
     also consumes two k steps per iteration. Other widths keep the shipped
     kernel (bit-identical either way, this only avoids an overrun). */
  if(tiling&&columns%8)tiling=0;
  /* The k-pair loop consumes two k steps per iteration. */
  if(tiling==2&&depth%2)tiling=1;
  /* Selector 3 reads an FP32 table, so it cannot silently demote to a kernel that
     indexes the table as __fp16 (that would read the wrong bytes). Its caller
     guarantees columns%8==0 and depth%2==0; anything else is a wiring bug and
     must fail loudly instead of returning wrong numbers. Partial row tiles are
     handled INSIDE the f32w kernel (project_part_f32w8 defers to the guarded
     project_part_f32w, still over the same fp32 table), so rows%8 needs no demotion. */
  if(tiling==3&&(depth%2||columns%8))return 1;
  /* The kernel writes all `rows` rows of every head; when rows==capacity no
     output element is left unwritten, so the zero fill can be skipped. */
  if(rows<capacity)for(size_t i=0;i<(size_t)heads*capacity*columns;i++)out[i]=0;
  MLAProjection work={out,input,weight,heads,capacity,rows,depth,columns,transposed};
#if MLA_FP16
  /* The decoder-only FP16 kernel indexes the table as `__fp16`, so it is never
     combined with selector 3: the Python selector clears the FP32 table whenever
     this branch can be taken (capacity==1). */
  if(rows==1&&!parallel){project_part_fp16(&work,0,1);return 0;}
#endif
  if(!parallel){(tiling==3?project_part_f32w8:tiling==2?project_part_kpair:tiling?project_part_blocked:project_part)(&work,0,1);return 0;}
  void *args[4]={&work,0,0,0};
  return ((MLAParallel)parallel)(tiling==3?project_worker_f32w:tiling==2?project_worker_kpair:tiling?project_worker_blocked:project_worker,args,0);
}
