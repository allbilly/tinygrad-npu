typedef __fp16 __fp164 __attribute__((aligned(2),ext_vector_type(4)));
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void r_640_4_64_4(unsigned char* restrict args_16) {
  __fp16 buf0[4];
  unsigned long val0 = (*(((unsigned long*)(args_16))+0));
  unsigned long val1 = (*(((unsigned long*)(args_16))+1));
  for (int Lidx1 = 0; Lidx1 < 640; Lidx1++) {
    *(buf0+0) = ((__fp16)(0.0f));
    *(buf0+1) = ((__fp16)(0.0f));
    *(buf0+2) = ((__fp16)(0.0f));
    *(buf0+3) = ((__fp16)(0.0f));
    for (int Lidx0 = 0; Lidx0 < 64; Lidx0++) {
      int alu4 = ((Lidx0<<2)+(Lidx1<<10));
      __fp164 val2 = (*((__fp164*)((((__fp16*)(val1))+(alu4+256)))));
      __fp164 val3 = (*((__fp164*)((((__fp16*)(val1))+(alu4+512)))));
      __fp164 val4 = (*((__fp164*)((((__fp16*)(val1))+(alu4+768)))));
      __fp164 val5 = (*((__fp164*)((((__fp16*)(val1))+alu4))));
      *(buf0+0) = ((*(buf0+0))+val5[0]+val5[1]+val5[2]+val5[3]);
      *(buf0+1) = ((*(buf0+1))+val2[0]+val2[1]+val2[2]+val2[3]);
      *(buf0+2) = ((*(buf0+2))+val3[0]+val3[1]+val3[2]+val3[3]);
      *(buf0+3) = ((*(buf0+3))+val4[0]+val4[1]+val4[2]+val4[3]);
    }
    *((float4*)((((float*)(val0))+(Lidx1<<2)))) = (float4){((float)((*(buf0+0)))),((float)((*(buf0+1)))),((float)((*(buf0+2)))),((float)((*(buf0+3))))};
  }
}
