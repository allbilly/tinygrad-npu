typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void E_131072_4(unsigned char* restrict args_8) {
  unsigned long val0 = (*(((unsigned long*)(args_8))+0));
  for (int Lidx0 = 0; Lidx0 < 131072; Lidx0++) {
    *((float4*)((((float*)(val0))+(Lidx0<<2)))) = (float4){0.0f,0.0f,0.0f,0.0f};
  }
}
