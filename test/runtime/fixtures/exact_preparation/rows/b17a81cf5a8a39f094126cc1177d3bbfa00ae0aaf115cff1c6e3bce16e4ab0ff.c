typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_128_384_4(float* restrict data0_128, float* restrict data1_196608) {
  float buf0[1];
  for (int Lidx1 = 0; Lidx1 < 128; Lidx1++) {
    *(buf0+0) = 0.0f;
    for (int Lidx0 = 0; Lidx0 < 384; Lidx0++) {
      float4 val0 = (*((float4*)((data1_196608+((Lidx0<<2)+(Lidx1*1536))))));
      *(buf0+0) = ((*(buf0+0))+(val0[0]*val0[0])+(val0[1]*val0[1])+(val0[2]*val0[2])+(val0[3]*val0[3]));
    }
    *(data0_128+Lidx1) = __builtin_sqrtf((((*(buf0+0))*0.0006510416860692203f)+9.999999974752427e-07f));
  }
}
