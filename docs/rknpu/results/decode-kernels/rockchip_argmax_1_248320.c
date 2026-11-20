typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
  typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));
  void rockchip_argmax_1_248320(void *args) {
    int *out = *(int **)args;
    const float *input = *((const float **)args + 1);
    for(int row=0;row<1;row++) {
      const float *data=input+row*248320;float4 best[4];int4 index[4];
      for(int b=0;b<4;b++){best[b]=(float4)(-__builtin_inff());index[b]=(int4){b*4,b*4+1,b*4+2,b*4+3};}
      int i=0;
      for(;i+15<248320;i+=16) {
        #pragma clang loop unroll(full)
        for(int b=0;b<4;b++){float4 value=*(const float4*)(data+i+b*4);int4 found=value>best[b];
          best[b]=found?value:best[b];index[b]=found?((int4){i+b*4,i+b*4+1,i+b*4+2,i+b*4+3}):index[b];}
      }
      float maximum=-__builtin_inff();int selected=0;
      for(int b=0;b<4;b++)for(int lane=0;lane<4;lane++){
        if(best[b][lane]>maximum || (best[b][lane]==maximum && index[b][lane]<selected)){
          maximum=best[b][lane];selected=index[b][lane];}
      }
      for(;i<248320;i++)if(data[i]>maximum){maximum=data[i];selected=i;}
      out[row]=selected;
    }
  }