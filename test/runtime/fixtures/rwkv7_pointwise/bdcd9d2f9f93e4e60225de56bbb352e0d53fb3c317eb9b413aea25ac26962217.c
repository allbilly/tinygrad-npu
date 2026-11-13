typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_toks_512_4(float* restrict data0_128, float* restrict data1_262144, float* restrict data2_128, const int toks_) {
  float buf0[1];
  for (int Lidx1 = 0; Lidx1 < toks_; Lidx1++) {
    float val0 = (*(data2_128+Lidx1));
    *(buf0+0) = 0.0f;
    for (int Lidx0 = 0; Lidx0 < 512; Lidx0++) {
      float4 val1 = (*((float4*)((data1_262144+((Lidx0<<2)+(Lidx1<<11))))));
      float alu1 = (val1[0]-val0);
      float alu2 = (val1[1]-val0);
      float alu3 = (val1[2]-val0);
      float alu4 = (val1[3]-val0);
      *(buf0+0) = ((*(buf0+0))+(alu1*alu1)+(alu2*alu2)+(alu3*alu3)+(alu4*alu4));
    }
    *(data0_128+Lidx1) = __builtin_sqrtf((((*(buf0+0))*0.00048828125f)+9.999999747378752e-06f));
  }
}
