typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_128_384_4(float* restrict data0_128, float* restrict data1_196608, float* restrict data2_196608, float* restrict data3_196608) {
  float buf0[1];
  for (int Lidx1 = 0; Lidx1 < 128; Lidx1++) {
    *(buf0+0) = 0.0f;
    for (int Lidx0 = 0; Lidx0 < 384; Lidx0++) {
      int alu1 = ((Lidx0<<2)+(Lidx1*1536));
      float4 val0 = (*((float4*)((data1_196608+alu1))));
      float4 val1 = (*((float4*)((data2_196608+alu1))));
      float4 val2 = (*((float4*)((data3_196608+alu1))));
      float alu2 = (val0[0]+val1[0]+val2[0]);
      float alu3 = (val0[1]+val1[1]+val2[1]);
      float alu4 = (val0[2]+val1[2]+val2[2]);
      float alu5 = (val0[3]+val1[3]+val2[3]);
      *(buf0+0) = ((*(buf0+0))+(alu2*alu2)+(alu3*alu3)+(alu4*alu4)+(alu5*alu5));
    }
    *(data0_128+Lidx1) = __builtin_sqrtf((((*(buf0+0))*0.0006510416860692203f)+9.999999974752427e-07f));
  }
}
