typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void E_128_512_4(float* restrict data0_262144, float* restrict data1_262144, const int toks_) {
  for (int Lidx0 = 0; Lidx0 < 128; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 512; Lidx1++) {
      int alu0 = ((Lidx0<<11)+(Lidx1<<2));
      float4 val0 = ((Lidx0<toks_)?*((float4*)((data1_262144+alu0))):(float4){0.0f,0.0f,0.0f,0.0f});
      *((float4*)((data0_262144+alu0))) = val0;
    }
  }
}
