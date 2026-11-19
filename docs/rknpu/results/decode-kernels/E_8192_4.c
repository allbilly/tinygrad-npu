
void E_8192_4(unsigned char* restrict args_32) {
  int val0 = (*(((int*)(args_32))+6));
  unsigned long val1 = (*(((unsigned long*)(args_32))+0));
  unsigned long val2 = (*(((unsigned long*)(args_32))+1));
  unsigned long val3 = (*(((unsigned long*)(args_32))+2));
  _Bool alu0 = (val0!=0);
  for (int Lidx1 = 0; Lidx1 < 8192; Lidx1++) {
    int alu1 = (Lidx1+16384);
    float val4 = (alu0?*(((float*)(val2))+alu1):0.0f);
    int alu2 = (Lidx1+8192);
    float val5 = (alu0?*(((float*)(val2))+alu2):0.0f);
    float val6 = (alu0?*(((float*)(val2))+Lidx1):0.0f);
    float val7 = (*(((float*)(val3))+Lidx1));
    *(((float*)(val1))+alu1) = val4;
    *(((float*)(val1))+(Lidx1+24576)) = val7;
    *(((float*)(val1))+alu2) = val5;
    *(((float*)(val1))+Lidx1) = val6;
  }
}
