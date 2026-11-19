typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void r_64_4(unsigned char* restrict args_16) {
  float buf0[1];
  unsigned long val0 = (*(((unsigned long*)(args_16))+1));
  *(buf0+0) = ((float)(-__builtin_inff()));
  for (int Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    float val1 = (*(buf0+0));
    float4 val2 = (*((float4*)((((float*)(val0))+(Lidx0<<2)))));
    float alu1 = ((val2[0]<val2[1])?val2[1]:val2[0]);
    float alu2 = ((alu1<val2[2])?val2[2]:alu1);
    float alu3 = ((alu2<val2[3])?val2[3]:alu2);
    float alu4 = ((val1<alu3)?alu3:val1);
    *(buf0+0) = alu4;
  }
  unsigned long val3 = (*(((unsigned long*)(args_16))+0));
  *(((float*)(val3))+0) = (*(buf0+0));
}
