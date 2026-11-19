typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void E_640_4(unsigned char* restrict args_32) {
  unsigned long val0 = (*(((unsigned long*)(args_32))+0));
  unsigned long val1 = (*(((unsigned long*)(args_32))+1));
  unsigned long val2 = (*(((unsigned long*)(args_32))+2));
  unsigned long val3 = (*(((unsigned long*)(args_32))+3));
  float val4 = (*(((float*)(val2))+0));
  for (int Lidx0 = 0; Lidx0 < 640; Lidx0++) {
    int alu0 = (Lidx0<<4);
    unsigned char val5 = (*(((unsigned char*)(val3))+(alu0+1)));
    unsigned char val6 = (*(((unsigned char*)(val3))+(alu0+10)));
    unsigned char val7 = (*(((unsigned char*)(val3))+(alu0+11)));
    unsigned char val8 = (*(((unsigned char*)(val3))+(alu0+12)));
    unsigned char val9 = (*(((unsigned char*)(val3))+(alu0+13)));
    unsigned char val10 = (*(((unsigned char*)(val3))+(alu0+14)));
    unsigned char val11 = (*(((unsigned char*)(val3))+(alu0+15)));
    unsigned char val12 = (*(((unsigned char*)(val3))+(alu0+2)));
    unsigned char val13 = (*(((unsigned char*)(val3))+(alu0+3)));
    unsigned char val14 = (*(((unsigned char*)(val3))+(alu0+4)));
    unsigned char val15 = (*(((unsigned char*)(val3))+(alu0+5)));
    unsigned char val16 = (*(((unsigned char*)(val3))+(alu0+6)));
    unsigned char val17 = (*(((unsigned char*)(val3))+(alu0+7)));
    unsigned char val18 = (*(((unsigned char*)(val3))+(alu0+8)));
    unsigned char val19 = (*(((unsigned char*)(val3))+(alu0+9)));
    unsigned char val20 = (*(((unsigned char*)(val3))+alu0));
    int alu1 = (Lidx0<<2);
    float4 val21 = (*((float4*)((((float*)(val1))+(alu1+325120)))));
    *((float4*)((((float*)(val0))+alu1))) = (float4){((val21[0]/val4)*((float)(((__fp16)(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(val20))+(((unsigned int)(val5))<<8u)+(((unsigned int)(val12))<<16u)+(((unsigned int)(val13))<<24u))))))))),((val21[1]/val4)*((float)(((__fp16)(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(val14))+(((unsigned int)(val15))<<8u)+(((unsigned int)(val16))<<16u)+(((unsigned int)(val17))<<24u))))))))),((val21[2]/val4)*((float)(((__fp16)(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(val18))+(((unsigned int)(val19))<<8u)+(((unsigned int)(val6))<<16u)+(((unsigned int)(val7))<<24u))))))))),((val21[3]/val4)*((float)(((__fp16)(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(val8))+(((unsigned int)(val9))<<8u)+(((unsigned int)(val10))<<16u)+(((unsigned int)(val11))<<24u)))))))))};
  }
}
