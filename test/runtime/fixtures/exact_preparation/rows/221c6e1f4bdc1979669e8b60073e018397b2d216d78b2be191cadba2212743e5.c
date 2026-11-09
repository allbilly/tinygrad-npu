typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_128_512_4(float* restrict data0_128, float* restrict data1_262144) {
  float buf0[1];
  for (int Lidx1 = 0; Lidx1 < 128; Lidx1++) {
    *(buf0+0) = 0.0f;
    for (int Lidx0 = 0; Lidx0 < 512; Lidx0++) {
      float4 val0 = (*((float4*)((data1_262144+((Lidx0<<2)+(Lidx1<<11))))));
      *(buf0+0) = ((*(buf0+0))+(val0[0]*val0[0])+(val0[1]*val0[1])+(val0[2]*val0[2])+(val0[3]*val0[3]));
    }
    *(data0_128+Lidx1) = ((*(buf0+0))*0.00048828125f);
  }
}
