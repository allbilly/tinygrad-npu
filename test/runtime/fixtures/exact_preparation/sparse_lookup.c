typedef __fp16 __fp164 __attribute__((aligned(8),ext_vector_type(4)));
void E_64_32_64_4(__fp16* restrict data0_524288, int* restrict data1_1, unsigned char* restrict data2_570425344) {
  int val0 = (*(data1_1+0));
  for (int Lidx1_0 = 0; Lidx1_0 < 64; Lidx1_0++) {
    int alu0 = ((val0*2176)+(Lidx1_0*34));
    for (int Lidx1_1 = 0; Lidx1_1 < 32; Lidx1_1++) {
      for (int Lidx2 = 0; Lidx2 < 64; Lidx2++) {
        int alu1 = (Lidx2<<12);
        _Bool alu2 = (val0<(alu1+1024));
        _Bool alu3 = (val0<(alu1+2048));
        _Bool alu4 = ((alu2!=1)&alu3);
        unsigned char val1 = (alu4?*(data2_570425344+(alu0+Lidx1_1+2)):((unsigned char)(0u)));
        _Bool alu5 = (val0<(alu1+3072));
        _Bool alu6 = ((alu3!=1)&alu5);
        unsigned char val2 = (alu6?*(data2_570425344+(alu0+Lidx1_1+2)):((unsigned char)(0u)));
        _Bool alu7 = ((alu5!=1)&(val0<(alu1+4096)));
        unsigned char val3 = (alu7?*(data2_570425344+(alu0+Lidx1_1+2)):((unsigned char)(0u)));
        _Bool alu8 = (((val0<alu1)!=1)&alu2);
        unsigned char val4 = (alu8?*(data2_570425344+(alu0+Lidx1_1+2)):((unsigned char)(0u)));
        unsigned char val5 = (alu4?*(data2_570425344+(alu0+1)):((unsigned char)(0u)));
        unsigned char val6 = (alu6?*(data2_570425344+(alu0+1)):((unsigned char)(0u)));
        unsigned char val7 = (alu7?*(data2_570425344+(alu0+1)):((unsigned char)(0u)));
        unsigned char val8 = (alu8?*(data2_570425344+(alu0+1)):((unsigned char)(0u)));
        unsigned char val9 = (alu4?*(data2_570425344+alu0):((unsigned char)(0u)));
        unsigned char val10 = (alu6?*(data2_570425344+alu0):((unsigned char)(0u)));
        unsigned char val11 = (alu7?*(data2_570425344+alu0):((unsigned char)(0u)));
        unsigned char val12 = (alu8?*(data2_570425344+alu0):((unsigned char)(0u)));
        __fp16 alu9 = (alu4?((__fp16)((((float)(__builtin_bit_cast(__fp16, (unsigned short)((((unsigned short)(val9))+(((unsigned short)(val5))<<((unsigned short)(8u))))))))*((float)(__builtin_bit_cast(signed char, (unsigned char)(val1))))))):((__fp16)(0.0f)));
        __fp16 alu10 = (alu6?((__fp16)((((float)(__builtin_bit_cast(__fp16, (unsigned short)((((unsigned short)(val10))+(((unsigned short)(val6))<<((unsigned short)(8u))))))))*((float)(__builtin_bit_cast(signed char, (unsigned char)(val2))))))):((__fp16)(0.0f)));
        __fp16 alu11 = (alu7?((__fp16)((((float)(__builtin_bit_cast(__fp16, (unsigned short)((((unsigned short)(val11))+(((unsigned short)(val7))<<((unsigned short)(8u))))))))*((float)(__builtin_bit_cast(signed char, (unsigned char)(val3))))))):((__fp16)(0.0f)));
        __fp16 alu12 = (alu8?((__fp16)((((float)(__builtin_bit_cast(__fp16, (unsigned short)((((unsigned short)(val12))+(((unsigned short)(val8))<<((unsigned short)(8u))))))))*((float)(__builtin_bit_cast(signed char, (unsigned char)(val4))))))):((__fp16)(0.0f)));
        *((__fp164*)((data0_524288+((Lidx1_0<<13)+(Lidx1_1<<8)+(Lidx2<<2))))) = (__fp164){alu12,alu9,alu10,alu11};
      }
    }
  }
}
