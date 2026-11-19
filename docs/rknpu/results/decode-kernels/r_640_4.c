typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void r_640_4(unsigned char* restrict args_16) {
  float buf0[1];
  unsigned long val0 = (*(((unsigned long*)(args_16))+1));
  *(buf0+0) = 0.0f;
  for (int Lidx0 = 0; Lidx0 < 640; Lidx0++) {
    float4 val1 = (*((float4*)((((float*)(val0))+((Lidx0<<2)+325120)))));
    *(buf0+0) = ((*(buf0+0))+(val1[0]*val1[0])+(val1[1]*val1[1])+(val1[2]*val1[2])+(val1[3]*val1[3]));
  }
  unsigned long val2 = (*(((unsigned long*)(args_16))+0));
  *(((float*)(val2))+0) = __builtin_sqrtf((((*(buf0+0))*0.0003906250058207661f)+9.999999974752427e-07f));
}
