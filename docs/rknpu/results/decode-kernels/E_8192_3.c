
void E_8192_3(unsigned char* restrict args_8) {
  unsigned long val0 = (*(((unsigned long*)(args_8))+0));
  for (int Lidx0 = 0; Lidx0 < 8192; Lidx0++) {
    int alu0 = (Lidx0*3);
    *(((float*)(val0))+(alu0+1)) = 0.0f;
    *(((float*)(val0))+(alu0+2)) = 0.0f;
    *(((float*)(val0))+alu0) = 0.0f;
  }
}
