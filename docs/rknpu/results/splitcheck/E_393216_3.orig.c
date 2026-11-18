
void E_393216_3(unsigned char* restrict args_24) {
  unsigned long val0 = (*(((unsigned long*)(args_24))+0));
  unsigned long val1 = (*(((unsigned long*)(args_24))+1));
  unsigned long val2 = (*(((unsigned long*)(args_24))+2));
  for (int Lidx0 = 0; Lidx0 < 393216; Lidx0++) {
    int alu0 = (Lidx0*3);
    int alu1 = (alu0+1);
    float val3 = (*(((float*)(val1))+alu1));
    int alu2 = (alu0+2);
    float val4 = (*(((float*)(val1))+alu2));
    float val5 = (*(((float*)(val1))+alu0));
    float val6 = (*(((float*)(val2))+alu1));
    float val7 = (*(((float*)(val2))+alu2));
    float val8 = (*(((float*)(val2))+alu0));
    float alu3 = (val3*-1.4426950216293335f);
    _Bool alu4 = (alu3!=alu3);
    float alu5 = ((alu3!=((float)(-__builtin_inff())))?alu3:0.0f);
    float alu6 = (alu4?0.0f:alu5);
    float alu7 = ((alu3!=((float)(__builtin_inff())))?alu6:0.0f);
    float alu8 = ((alu7<0.0f)?-0.5f:0.5f);
    int cast0 = ((int)((alu7+alu8)));
    float alu9 = (val4*-1.4426950216293335f);
    _Bool alu10 = (alu9!=alu9);
    float alu11 = ((alu9!=((float)(-__builtin_inff())))?alu9:0.0f);
    float alu12 = (alu10?0.0f:alu11);
    float alu13 = ((alu9!=((float)(__builtin_inff())))?alu12:0.0f);
    float alu14 = ((alu13<0.0f)?-0.5f:0.5f);
    int cast1 = ((int)((alu13+alu14)));
    float alu15 = (val5*-1.4426950216293335f);
    _Bool alu16 = (alu15!=alu15);
    float alu17 = ((alu15!=((float)(-__builtin_inff())))?alu15:0.0f);
    float alu18 = (alu16?0.0f:alu17);
    float alu19 = ((alu15!=((float)(__builtin_inff())))?alu18:0.0f);
    float alu20 = ((alu19<0.0f)?-0.5f:0.5f);
    int cast2 = ((int)((alu19+alu20)));
    int alu21 = (cast0>>1);
    int alu22 = (cast1>>1);
    int alu23 = (cast2>>1);
    float alu24 = (alu7-((float)(cast0)));
    float alu25 = (((alu3<128.0f)==0)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu24)+0.0013392627006396651f)*alu24)+0.009618384763598442f)*alu24)+0.055503472685813904f)*alu24)+0.24022644758224487f)*alu24)+0.6931471824645996f)*alu24)+1.0f)*__builtin_bit_cast(float, (int)(((alu21+127)<<23)))*__builtin_bit_cast(float, (int)((((cast0-alu21)+127)<<23)))));
    float alu26 = ((alu3<-150.0f)?0.0f:alu25);
    float alu27 = (alu4?((float)(__builtin_nanf(""))):alu26);
    float alu28 = (alu13-((float)(cast1)));
    float alu29 = (((alu9<128.0f)==0)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu28)+0.0013392627006396651f)*alu28)+0.009618384763598442f)*alu28)+0.055503472685813904f)*alu28)+0.24022644758224487f)*alu28)+0.6931471824645996f)*alu28)+1.0f)*__builtin_bit_cast(float, (int)(((alu22+127)<<23)))*__builtin_bit_cast(float, (int)((((cast1-alu22)+127)<<23)))));
    float alu30 = ((alu9<-150.0f)?0.0f:alu29);
    float alu31 = (alu10?((float)(__builtin_nanf(""))):alu30);
    float alu32 = (alu19-((float)(cast2)));
    float alu33 = (((alu15<128.0f)==0)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu32)+0.0013392627006396651f)*alu32)+0.009618384763598442f)*alu32)+0.055503472685813904f)*alu32)+0.24022644758224487f)*alu32)+0.6931471824645996f)*alu32)+1.0f)*__builtin_bit_cast(float, (int)(((alu23+127)<<23)))*__builtin_bit_cast(float, (int)((((cast2-alu23)+127)<<23)))));
    float alu34 = ((alu15<-150.0f)?0.0f:alu33);
    float alu35 = (alu16?((float)(__builtin_nanf(""))):alu34);
    *(((float*)(val0))+alu1) = ((val3/(1.0f+alu27))*val6);
    *(((float*)(val0))+alu2) = ((val4/(1.0f+alu31))*val7);
    *(((float*)(val0))+alu0) = ((val5/(1.0f+alu35))*val8);
  }
}
