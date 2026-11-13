
void E_2_toks_2048_3(float* restrict data0_1572864, float* restrict data1_262144, float* restrict data2_264192, __fp16* restrict data3_12288, const int toks_) {
  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < toks_; Lidx1++) {
      for (int Lidx2 = 0; Lidx2 < 2048; Lidx2++) {
        int alu0 = ((Lidx0*6144)+Lidx2);
        __fp16 val0 = (*(data3_12288+(alu0+2048)));
        __fp16 val1 = (*(data3_12288+(alu0+4096)));
        __fp16 val2 = (*(data3_12288+alu0));
        int alu1 = ((Lidx1<<11)+Lidx2);
        float val3 = (*(data1_262144+alu1));
        float val4 = (*(data2_264192+alu1));
        int alu2 = (alu1+(Lidx0*786432));
        float alu3 = (val4-val3);
        *(data0_1572864+(alu2+262144)) = (val3+(alu3*((float)(val0))));
        *(data0_1572864+(alu2+524288)) = (val3+(alu3*((float)(val1))));
        *(data0_1572864+alu2) = (val3+(alu3*((float)(val2))));
      }
    }
  }
}
