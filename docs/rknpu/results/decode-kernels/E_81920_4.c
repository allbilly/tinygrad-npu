typedef __fp16 __fp164 __attribute__((aligned(2),ext_vector_type(4)));
void E_81920_4(unsigned char* restrict args_8) {
  unsigned long val0 = (*(((unsigned long*)(args_8))+0));
  for (int Lidx0 = 0; Lidx0 < 81920; Lidx0++) {
    *((__fp164*)((((__fp16*)(val0))+(Lidx0<<2)))) = (__fp164){((__fp16)(0.0f)),((__fp16)(0.0f)),((__fp16)(0.0f)),((__fp16)(0.0f))};
  }
}
