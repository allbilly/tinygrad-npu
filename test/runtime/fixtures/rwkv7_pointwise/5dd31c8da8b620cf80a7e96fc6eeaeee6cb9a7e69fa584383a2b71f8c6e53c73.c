typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void E_toks_512_4(float* restrict data0_262144, float* restrict data1_262144, float* restrict data2_262144, float* restrict data3_128, const int toks_) {
  for (int Lidx0 = 0; Lidx0 < toks_; Lidx0++) {
    float val0 = (*(data3_128+Lidx0));
    for (int Lidx1 = 0; Lidx1 < 512; Lidx1++) {
      int alu0 = ((Lidx0<<11)+(Lidx1<<2));
      float4 val1 = (*((float4*)((data1_262144+alu0))));
      float4 val2 = (*((float4*)((data2_262144+alu0))));
      *((float4*)((data0_262144+alu0))) = (float4){((val1[0]+val2[0])-val0),((val1[1]+val2[1])-val0),((val1[2]+val2[2])-val0),((val1[3]+val2[3])-val0)};
    }
  }
}
