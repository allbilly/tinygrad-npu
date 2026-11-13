typedef __fp16 __fp164 __attribute__((aligned(8),ext_vector_type(4)));
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void E_toks_512_4(float* restrict data0_262144, float* restrict data1_262144, float* restrict data2_128, __fp16* restrict data3_2048, __fp16* restrict data4_2048, const int toks_) {
  for (int Lidx0 = 0; Lidx0 < toks_; Lidx0++) {
    float val0 = (*(data2_128+Lidx0));
    for (int Lidx1 = 0; Lidx1 < 512; Lidx1++) {
      int alu0 = (Lidx1<<2);
      __fp164 val1 = (*((__fp164*)((data3_2048+alu0))));
      __fp164 val2 = (*((__fp164*)((data4_2048+alu0))));
      int alu1 = ((Lidx0<<11)+alu0);
      float4 val3 = (*((float4*)((data1_262144+alu1))));
      *((float4*)((data0_262144+alu1))) = (float4){(((val3[0]/val0)*((float)(val1[0])))+((float)(val2[0]))),(((val3[1]/val0)*((float)(val1[1])))+((float)(val2[1]))),(((val3[2]/val0)*((float)(val1[2])))+((float)(val2[2]))),(((val3[3]/val0)*((float)(val1[3])))+((float)(val2[3])))};
    }
  }
}
