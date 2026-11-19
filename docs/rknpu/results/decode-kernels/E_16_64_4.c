typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void E_16_64_4(unsigned char* restrict args_24) {
  unsigned long val0 = (*(((unsigned long*)(args_24))+0));
  unsigned long val1 = (*(((unsigned long*)(args_24))+1));
  unsigned long val2 = (*(((unsigned long*)(args_24))+2));
  for (int Lidx0_0 = 0; Lidx0_0 < 16; Lidx0_0++) {
    for (int Lidx0_1 = 0; Lidx0_1 < 64; Lidx0_1++) {
      int alu0 = (Lidx0_1<<2);
      int alu1 = ((Lidx0_0<<8)+alu0);
      float4 val3 = (*((float4*)((((float*)(val1))+alu1))));
      float4 val4 = (*((float4*)((((float*)(val2))+((Lidx0_0<<9)+alu0+256)))));
      float alu2 = (val4[0]*-1.4426950216293335f);
      _Bool alu3 = (alu2!=alu2);
      float alu4 = ((alu2!=((float)(-__builtin_inff())))?alu2:0.0f);
      float alu5 = (alu3?0.0f:alu4);
      float alu6 = ((alu2!=((float)(__builtin_inff())))?alu5:0.0f);
      float alu7 = ((alu6<0.0f)?-0.5f:0.5f);
      int cast0 = ((int)((alu6+alu7)));
      float alu8 = (val4[1]*-1.4426950216293335f);
      _Bool alu9 = (alu8!=alu8);
      float alu10 = ((alu8!=((float)(-__builtin_inff())))?alu8:0.0f);
      float alu11 = (alu9?0.0f:alu10);
      float alu12 = ((alu8!=((float)(__builtin_inff())))?alu11:0.0f);
      float alu13 = ((alu12<0.0f)?-0.5f:0.5f);
      int cast1 = ((int)((alu12+alu13)));
      float alu14 = (val4[2]*-1.4426950216293335f);
      _Bool alu15 = (alu14!=alu14);
      float alu16 = ((alu14!=((float)(-__builtin_inff())))?alu14:0.0f);
      float alu17 = (alu15?0.0f:alu16);
      float alu18 = ((alu14!=((float)(__builtin_inff())))?alu17:0.0f);
      float alu19 = ((alu18<0.0f)?-0.5f:0.5f);
      int cast2 = ((int)((alu18+alu19)));
      float alu20 = (val4[3]*-1.4426950216293335f);
      _Bool alu21 = (alu20!=alu20);
      float alu22 = ((alu20!=((float)(-__builtin_inff())))?alu20:0.0f);
      float alu23 = (alu21?0.0f:alu22);
      float alu24 = ((alu20!=((float)(__builtin_inff())))?alu23:0.0f);
      float alu25 = ((alu24<0.0f)?-0.5f:0.5f);
      int cast3 = ((int)((alu24+alu25)));
      int alu26 = (cast0>>1);
      int alu27 = (cast1>>1);
      int alu28 = (cast2>>1);
      int alu29 = (cast3>>1);
      float alu30 = (alu6-((float)(cast0)));
      float alu31 = (((alu2<128.0f)==0)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu30)+0.0013392627006396651f)*alu30)+0.009618384763598442f)*alu30)+0.055503472685813904f)*alu30)+0.24022644758224487f)*alu30)+0.6931471824645996f)*alu30)+1.0f)*__builtin_bit_cast(float, (int)(((alu26+127)<<23)))*__builtin_bit_cast(float, (int)((((cast0-alu26)+127)<<23)))));
      float alu32 = ((alu2<-150.0f)?0.0f:alu31);
      float alu33 = (alu3?((float)(__builtin_nanf(""))):alu32);
      float alu34 = (alu12-((float)(cast1)));
      float alu35 = (((alu8<128.0f)==0)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu34)+0.0013392627006396651f)*alu34)+0.009618384763598442f)*alu34)+0.055503472685813904f)*alu34)+0.24022644758224487f)*alu34)+0.6931471824645996f)*alu34)+1.0f)*__builtin_bit_cast(float, (int)(((alu27+127)<<23)))*__builtin_bit_cast(float, (int)((((cast1-alu27)+127)<<23)))));
      float alu36 = ((alu8<-150.0f)?0.0f:alu35);
      float alu37 = (alu9?((float)(__builtin_nanf(""))):alu36);
      float alu38 = (alu18-((float)(cast2)));
      float alu39 = (((alu14<128.0f)==0)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu38)+0.0013392627006396651f)*alu38)+0.009618384763598442f)*alu38)+0.055503472685813904f)*alu38)+0.24022644758224487f)*alu38)+0.6931471824645996f)*alu38)+1.0f)*__builtin_bit_cast(float, (int)(((alu28+127)<<23)))*__builtin_bit_cast(float, (int)((((cast2-alu28)+127)<<23)))));
      float alu40 = ((alu14<-150.0f)?0.0f:alu39);
      float alu41 = (alu15?((float)(__builtin_nanf(""))):alu40);
      float alu42 = (alu24-((float)(cast3)));
      float alu43 = (((alu20<128.0f)==0)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu42)+0.0013392627006396651f)*alu42)+0.009618384763598442f)*alu42)+0.055503472685813904f)*alu42)+0.24022644758224487f)*alu42)+0.6931471824645996f)*alu42)+1.0f)*__builtin_bit_cast(float, (int)(((alu29+127)<<23)))*__builtin_bit_cast(float, (int)((((cast3-alu29)+127)<<23)))));
      float alu44 = ((alu20<-150.0f)?0.0f:alu43);
      float alu45 = (alu21?((float)(__builtin_nanf(""))):alu44);
      *((float4*)((((float*)(val0))+alu1))) = (float4){(val3[0]/(1.0f+alu33)),(val3[1]/(1.0f+alu37)),(val3[2]/(1.0f+alu41)),(val3[3]/(1.0f+alu45))};
    }
  }
}
