typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_8_128_128_4(float* restrict data0_1024, float* restrict data1_655360) {
  float buf0[1];
  for (int Lidx1 = 0; Lidx1 < 8; Lidx1++) {
    for (int Lidx2 = 0; Lidx2 < 128; Lidx2++) {
      *(buf0+0) = 0.0f;
      for (int Lidx0 = 0; Lidx0 < 128; Lidx0++) {
        float4 val0 = (*((float4*)((data1_655360+((Lidx0<<2)+(Lidx1<<9)+(Lidx2*5120))))));
        *(buf0+0) = ((*(buf0+0))+(val0[0]*val0[0])+(val0[1]*val0[1])+(val0[2]*val0[2])+(val0[3]*val0[3]));
      }
      *(data0_1024+((Lidx1<<7)+Lidx2)) = __builtin_sqrtf((((*(buf0+0))*0.001953125f)+9.999999974752427e-07f));
    }
  }
}
