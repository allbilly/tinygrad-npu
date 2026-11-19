typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void r_64_4_970(unsigned char* restrict args_16) {
  float buf0[4];
  unsigned long val0 = (*(((unsigned long*)(args_16))+0));
  unsigned long val1 = (*(((unsigned long*)(args_16))+1));
  for (int Lidx1 = 0; Lidx1 < 64; Lidx1++) {
    *(buf0+0) = ((float)(-__builtin_inff()));
    *(buf0+1) = ((float)(-__builtin_inff()));
    *(buf0+2) = ((float)(-__builtin_inff()));
    *(buf0+3) = ((float)(-__builtin_inff()));
    for (int Lidx0 = 0; Lidx0 < 970; Lidx0++) {
      int alu4 = ((Lidx1*3880)+Lidx0);
      float val2 = (*(((float*)(val1))+(alu4+1940)));
      float val3 = (*(((float*)(val1))+(alu4+2910)));
      float val4 = (*(((float*)(val1))+(alu4+970)));
      float val5 = (*(((float*)(val1))+alu4));
      float val6 = (*(buf0+0));
      float val7 = (*(buf0+1));
      float val8 = (*(buf0+2));
      float val9 = (*(buf0+3));
      float alu5 = ((val6<val5)?val5:val6);
      float alu6 = ((val7<val4)?val4:val7);
      float alu7 = ((val8<val2)?val2:val8);
      float alu8 = ((val9<val3)?val3:val9);
      *(buf0+0) = alu5;
      *(buf0+1) = alu6;
      *(buf0+2) = alu7;
      *(buf0+3) = alu8;
    }
    *((float4*)((((float*)(val0))+(Lidx1<<2)))) = (float4){(*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))};
  }
}
