/* Same scalar float32 sigmoid legalization as the CPU renderer; distribute rows only. */
#include <stddef.h>
extern void *memset(void *,int,size_t);
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
static void sigmoid_part(float* restrict data0_786432, const float* restrict data1_1572864, int toks_, int depth, int group, int groups) {
  for (int Lidx0 = group; Lidx0 < toks_; Lidx0 += groups) {
    for (int Lidx1 = 0; Lidx1 < depth/4; Lidx1++) {
      int alu0 = (Lidx1<<2);
      float4 val0 = (*((float4*)((data1_1572864+((Lidx0*(depth*2))+alu0)))));
      float alu1 = (val0[0]*-1.4426950216293335f);
      _Bool alu2 = (alu1!=alu1);
      float alu3 = ((alu1!=((float)(-__builtin_inff())))?alu1:0.0f);
      float alu4 = (alu2?0.0f:alu3);
      float alu5 = ((alu1!=((float)(__builtin_inff())))?alu4:0.0f);
      float alu6 = ((alu5<0.0f)?-0.5f:0.5f);
      int cast0 = ((int)((alu5+alu6)));
      float alu7 = (val0[1]*-1.4426950216293335f);
      _Bool alu8 = (alu7!=alu7);
      float alu9 = ((alu7!=((float)(-__builtin_inff())))?alu7:0.0f);
      float alu10 = (alu8?0.0f:alu9);
      float alu11 = ((alu7!=((float)(__builtin_inff())))?alu10:0.0f);
      float alu12 = ((alu11<0.0f)?-0.5f:0.5f);
      int cast1 = ((int)((alu11+alu12)));
      float alu13 = (val0[2]*-1.4426950216293335f);
      _Bool alu14 = (alu13!=alu13);
      float alu15 = ((alu13!=((float)(-__builtin_inff())))?alu13:0.0f);
      float alu16 = (alu14?0.0f:alu15);
      float alu17 = ((alu13!=((float)(__builtin_inff())))?alu16:0.0f);
      float alu18 = ((alu17<0.0f)?-0.5f:0.5f);
      int cast2 = ((int)((alu17+alu18)));
      float alu19 = (val0[3]*-1.4426950216293335f);
      _Bool alu20 = (alu19!=alu19);
      float alu21 = ((alu19!=((float)(-__builtin_inff())))?alu19:0.0f);
      float alu22 = (alu20?0.0f:alu21);
      float alu23 = ((alu19!=((float)(__builtin_inff())))?alu22:0.0f);
      float alu24 = ((alu23<0.0f)?-0.5f:0.5f);
      int cast3 = ((int)((alu23+alu24)));
      int alu25 = (cast0>>1);
      int alu26 = (cast1>>1);
      int alu27 = (cast2>>1);
      int alu28 = (cast3>>1);
      float alu29 = (alu5-((float)(cast0)));
      float alu30 = (((alu1<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu29)+0.0013392627006396651f)*alu29)+0.009618384763598442f)*alu29)+0.055503472685813904f)*alu29)+0.24022644758224487f)*alu29)+0.6931471824645996f)*alu29)+1.0f)*__builtin_bit_cast(float, (int)(((alu25+127)<<23)))*__builtin_bit_cast(float, (int)((((cast0-alu25)+127)<<23)))));
      float alu31 = ((alu1<-150.0f)?0.0f:alu30);
      float alu32 = (alu2?((float)(__builtin_nanf(""))):alu31);
      float alu33 = (alu11-((float)(cast1)));
      float alu34 = (((alu7<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu33)+0.0013392627006396651f)*alu33)+0.009618384763598442f)*alu33)+0.055503472685813904f)*alu33)+0.24022644758224487f)*alu33)+0.6931471824645996f)*alu33)+1.0f)*__builtin_bit_cast(float, (int)(((alu26+127)<<23)))*__builtin_bit_cast(float, (int)((((cast1-alu26)+127)<<23)))));
      float alu35 = ((alu7<-150.0f)?0.0f:alu34);
      float alu36 = (alu8?((float)(__builtin_nanf(""))):alu35);
      float alu37 = (alu17-((float)(cast2)));
      float alu38 = (((alu13<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu37)+0.0013392627006396651f)*alu37)+0.009618384763598442f)*alu37)+0.055503472685813904f)*alu37)+0.24022644758224487f)*alu37)+0.6931471824645996f)*alu37)+1.0f)*__builtin_bit_cast(float, (int)(((alu27+127)<<23)))*__builtin_bit_cast(float, (int)((((cast2-alu27)+127)<<23)))));
      float alu39 = ((alu13<-150.0f)?0.0f:alu38);
      float alu40 = (alu14?((float)(__builtin_nanf(""))):alu39);
      float alu41 = (alu23-((float)(cast3)));
      float alu42 = (((alu19<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu41)+0.0013392627006396651f)*alu41)+0.009618384763598442f)*alu41)+0.055503472685813904f)*alu41)+0.24022644758224487f)*alu41)+0.6931471824645996f)*alu41)+1.0f)*__builtin_bit_cast(float, (int)(((alu28+127)<<23)))*__builtin_bit_cast(float, (int)((((cast3-alu28)+127)<<23)))));
      float alu43 = ((alu19<-150.0f)?0.0f:alu42);
      float alu44 = (alu20?((float)(__builtin_nanf(""))):alu43);
      *((float4*)((data0_786432+((Lidx0*depth)+alu0)))) = (float4){(1.0f/(1.0f+alu32)),(1.0f/(1.0f+alu36)),(1.0f/(1.0f+alu40)),(1.0f/(1.0f+alu44))};
    }
  }
}

typedef int (*sigmoid_parallel_fn)(void (*)(void **,int,int),void **,int);
typedef struct {float *out;const float *input;int rows,depth,done[4];} SigmoidWork;
static void sigmoid_worker(void **args,int unused,int group) {
  (void)unused;
  SigmoidWork *w=(SigmoidWork*)args[0];
  if(group<0||group>=4)return;
  sigmoid_part(w->out,w->input,w->rows,w->depth,group,4);
  w->done[group]=1;
}
static int sigmoid_gate(float *out,const float *input,int rows,int capacity,int depth,sigmoid_parallel_fn parallel) {
  if(!out||!input||rows<1||rows>capacity||capacity>128||depth<4||depth>32768||depth%4)return 1;
  if(!parallel||rows<4)sigmoid_part(out,input,rows,depth,0,1);
  else {
    SigmoidWork work={.out=out,.input=input,.rows=rows,.depth=depth,.done={0,0,0,0}};
    void *args[4]={&work,0,0,0};
    int error=parallel(sigmoid_worker,args,0),completed=0;
    for(int i=0;i<4;i++)completed+=work.done[i]!=0;
    if(completed==0&&error)sigmoid_part(out,input,rows,depth,0,1);
    else if(error||completed!=4)return error?error:2;
  }
  memset(out+(size_t)rows*depth,0,(size_t)(capacity-rows)*depth*sizeof(float));
  return 0;
}
