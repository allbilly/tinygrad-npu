typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_toks_512_4(float* restrict data0_128, float* restrict data1_262144, float* restrict data2_262144, const int toks_) {
  float buf0[1];
  for (int Lidx1 = 0; Lidx1 < toks_; Lidx1++) {
    *(buf0+0) = 0.0f;
    for (int Lidx0 = 0; Lidx0 < 512; Lidx0++) {
      int alu1 = ((Lidx0<<2)+(Lidx1<<11));
      float4 val0 = (*((float4*)((data1_262144+alu1))));
      float4 val1 = (*((float4*)((data2_262144+alu1))));
      *(buf0+0) = ((*(buf0+0))+val0[0]+val1[0]+val0[1]+val1[1]+val0[2]+val1[2]+val0[3]+val1[3]);
    }
    *(data0_128+Lidx1) = ((*(buf0+0))*0.00048828125f);
  }
}
