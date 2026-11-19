typedef __fp16 __fp164 __attribute__((aligned(2),ext_vector_type(4)));
void E_80_32_64_4(unsigned char* restrict args_24) {
  unsigned long val0 = (*(((unsigned long*)(args_24))+1));
  int val1 = (*(((int*)(val0))+0));
  unsigned long val2 = (*(((unsigned long*)(args_24))+0));
  unsigned long val3 = (*(((unsigned long*)(args_24))+2));
  for (int Lidx1_0 = 0; Lidx1_0 < 80; Lidx1_0++) {
    int alu0 = ((val1*2720)+(Lidx1_0*34));
    for (int Lidx1_1 = 0; Lidx1_1 < 32; Lidx1_1++) {
      for (int Lidx2 = 0; Lidx2 < 64; Lidx2++) {
        int alu1 = (Lidx2*3880);
        _Bool alu2 = (val1<(alu1+1940));
        _Bool alu3 = (val1<(alu1+2910));
        _Bool alu4 = ((alu2==0)&alu3);
        unsigned char val4 = (alu4?*(((unsigned char*)(val3))+(alu0+Lidx1_1+2)):((unsigned char)(0u)));
        _Bool alu5 = ((alu3==0)&(val1<(alu1+3880)));
        unsigned char val5 = (alu5?*(((unsigned char*)(val3))+(alu0+Lidx1_1+2)):((unsigned char)(0u)));
        _Bool alu6 = (val1<(alu1+970));
        _Bool alu7 = ((alu6==0)&alu2);
        unsigned char val6 = (alu7?*(((unsigned char*)(val3))+(alu0+Lidx1_1+2)):((unsigned char)(0u)));
        _Bool alu8 = (((val1<alu1)==0)&alu6);
        unsigned char val7 = (alu8?*(((unsigned char*)(val3))+(alu0+Lidx1_1+2)):((unsigned char)(0u)));
        unsigned char val8 = (alu4?*(((unsigned char*)(val3))+(alu0+1)):((unsigned char)(0u)));
        unsigned char val9 = (alu5?*(((unsigned char*)(val3))+(alu0+1)):((unsigned char)(0u)));
        unsigned char val10 = (alu7?*(((unsigned char*)(val3))+(alu0+1)):((unsigned char)(0u)));
        unsigned char val11 = (alu8?*(((unsigned char*)(val3))+(alu0+1)):((unsigned char)(0u)));
        unsigned char val12 = (alu4?*(((unsigned char*)(val3))+alu0):((unsigned char)(0u)));
        unsigned char val13 = (alu5?*(((unsigned char*)(val3))+alu0):((unsigned char)(0u)));
        unsigned char val14 = (alu7?*(((unsigned char*)(val3))+alu0):((unsigned char)(0u)));
        unsigned char val15 = (alu8?*(((unsigned char*)(val3))+alu0):((unsigned char)(0u)));
        __fp16 alu9 = (alu4?((__fp16)((((float)(__builtin_bit_cast(__fp16, (unsigned short)(((unsigned short)((((unsigned int)(((unsigned short)(val12))))+((unsigned int)(((unsigned short)((((unsigned int)(((unsigned short)(val8))))<<8u))))))))))))*((float)(__builtin_bit_cast(signed char, (unsigned char)(val4))))))):((__fp16)(0.0f)));
        __fp16 alu10 = (alu5?((__fp16)((((float)(__builtin_bit_cast(__fp16, (unsigned short)(((unsigned short)((((unsigned int)(((unsigned short)(val13))))+((unsigned int)(((unsigned short)((((unsigned int)(((unsigned short)(val9))))<<8u))))))))))))*((float)(__builtin_bit_cast(signed char, (unsigned char)(val5))))))):((__fp16)(0.0f)));
        __fp16 alu11 = (alu7?((__fp16)((((float)(__builtin_bit_cast(__fp16, (unsigned short)(((unsigned short)((((unsigned int)(((unsigned short)(val14))))+((unsigned int)(((unsigned short)((((unsigned int)(((unsigned short)(val10))))<<8u))))))))))))*((float)(__builtin_bit_cast(signed char, (unsigned char)(val6))))))):((__fp16)(0.0f)));
        __fp16 alu12 = (alu8?((__fp16)((((float)(__builtin_bit_cast(__fp16, (unsigned short)(((unsigned short)((((unsigned int)(((unsigned short)(val15))))+((unsigned int)(((unsigned short)((((unsigned int)(((unsigned short)(val11))))<<8u))))))))))))*((float)(__builtin_bit_cast(signed char, (unsigned char)(val7))))))):((__fp16)(0.0f)));
        *((__fp164*)((((__fp16*)(val2))+((Lidx1_0<<13)+(Lidx1_1<<8)+(Lidx2<<2))))) = (__fp164){alu12,alu11,alu9,alu10};
      }
    }
  }
}
