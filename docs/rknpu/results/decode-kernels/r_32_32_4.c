typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void r_32_32_4(unsigned char* restrict args_16) {
  float buf0[1];
  unsigned long val0 = (*(((unsigned long*)(args_16))+0));
  unsigned long val1 = (*(((unsigned long*)(args_16))+1));
  for (int Lidx1 = 0; Lidx1 < 32; Lidx1++) {
    *(buf0+0) = 0.0f;
    for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {
      float4 val2 = (*((float4*)((((float*)(val1))+((Lidx0<<2)+(Lidx1<<7))))));
      *(buf0+0) = ((*(buf0+0))+(val2[0]*val2[0])+(val2[1]*val2[1])+(val2[2]*val2[2])+(val2[3]*val2[3]));
    }
    *(((float*)(val0))+Lidx1) = __builtin_sqrtf((((*(buf0+0))*0.0078125f)+9.999999974752427e-07f));
  }
}
