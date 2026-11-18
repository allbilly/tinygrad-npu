
static void E_32_32_128_4_part(float* restrict data0_524288, float* restrict data1_1073152, unsigned char* restrict data2_131072,int group) {
  for (int Lidx0 = 32*group/4; Lidx0 < 32*(group+1)/4; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 32; Lidx1++) {
      for (int Lidx2 = 0; Lidx2 < 128; Lidx2++) {
        int alu0 = ((Lidx0<<11)+(Lidx2<<4));
        unsigned char val0 = (*(data2_131072+(alu0+65536)));
        unsigned char val1 = (*(data2_131072+(alu0+65537)));
        unsigned char val2 = (*(data2_131072+(alu0+65538)));
        unsigned char val3 = (*(data2_131072+(alu0+65539)));
        unsigned char val4 = (*(data2_131072+(alu0+65540)));
        unsigned char val5 = (*(data2_131072+(alu0+65541)));
        unsigned char val6 = (*(data2_131072+(alu0+65542)));
        unsigned char val7 = (*(data2_131072+(alu0+65543)));
        unsigned char val8 = (*(data2_131072+(alu0+65544)));
        unsigned char val9 = (*(data2_131072+(alu0+65545)));
        unsigned char val10 = (*(data2_131072+(alu0+65546)));
        unsigned char val11 = (*(data2_131072+(alu0+65547)));
        unsigned char val12 = (*(data2_131072+(alu0+65548)));
        unsigned char val13 = (*(data2_131072+(alu0+65549)));
        unsigned char val14 = (*(data2_131072+(alu0+65550)));
        unsigned char val15 = (*(data2_131072+(alu0+65551)));
        int alu1 = ((Lidx0<<7)+Lidx2+(Lidx1<<15));
        float val16 = (*(data1_1073152+(alu1+12288)));
        float val17 = (*(data1_1073152+(alu1+20480)));
        float val18 = (*(data1_1073152+(alu1+28672)));
        float val19 = (*(data1_1073152+(alu1+36864)));
        float val20 = (*(data1_1073152+(alu1+4096)));
        float val21 = (*(data1_1073152+(alu1+45056)));
        float val22 = (*(data1_1073152+(alu1+53248)));
        int alu2 = ((Lidx1<<9)+Lidx2+(Lidx0<<14));
        float cast0 = ((float)(((__fp16)(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(val0))+(((unsigned int)(val1))<<8u)+(((unsigned int)(val2))<<16u)+(((unsigned int)(val3))<<24u))))))));
        float cast1 = ((float)(((__fp16)(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(val4))+(((unsigned int)(val5))<<8u)+(((unsigned int)(val6))<<16u)+(((unsigned int)(val7))<<24u))))))));
        float cast2 = ((float)(((__fp16)(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(val8))+(((unsigned int)(val9))<<8u)+(((unsigned int)(val10))<<16u)+(((unsigned int)(val11))<<24u))))))));
        float cast3 = ((float)(((__fp16)(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(val12))+(((unsigned int)(val13))<<8u)+(((unsigned int)(val14))<<16u)+(((unsigned int)(val15))<<24u))))))));
        float alu3 = ((val16*cast0)+(val17*cast1)+(val18*cast2)+(val19*cast3));
        float alu4 = (alu3*-1.4426950216293335f);
        _Bool alu5 = (alu4!=alu4);
        float alu6 = ((alu4!=((float)(-__builtin_inff())))?alu4:0.0f);
        float alu7 = (alu5?0.0f:alu6);
        float alu8 = ((alu4!=((float)(__builtin_inff())))?alu7:0.0f);
        float alu9 = ((alu8<0.0f)?-0.5f:0.5f);
        int cast4 = ((int)((alu8+alu9)));
        float alu10 = ((val17*cast0)+(val18*cast1)+(val19*cast2)+(val21*cast3));
        float alu11 = (alu10*-1.4426950216293335f);
        _Bool alu12 = (alu11!=alu11);
        float alu13 = ((alu11!=((float)(-__builtin_inff())))?alu11:0.0f);
        float alu14 = (alu12?0.0f:alu13);
        float alu15 = ((alu11!=((float)(__builtin_inff())))?alu14:0.0f);
        float alu16 = ((alu15<0.0f)?-0.5f:0.5f);
        int cast5 = ((int)((alu15+alu16)));
        float alu17 = ((val18*cast0)+(val19*cast1)+(val21*cast2)+(val22*cast3));
        float alu18 = (alu17*-1.4426950216293335f);
        _Bool alu19 = (alu18!=alu18);
        float alu20 = ((alu18!=((float)(-__builtin_inff())))?alu18:0.0f);
        float alu21 = (alu19?0.0f:alu20);
        float alu22 = ((alu18!=((float)(__builtin_inff())))?alu21:0.0f);
        float alu23 = ((alu22<0.0f)?-0.5f:0.5f);
        int cast6 = ((int)((alu22+alu23)));
        float alu24 = ((val20*cast0)+(val16*cast1)+(val17*cast2)+(val18*cast3));
        float alu25 = (alu24*-1.4426950216293335f);
        _Bool alu26 = (alu25!=alu25);
        float alu27 = ((alu25!=((float)(-__builtin_inff())))?alu25:0.0f);
        float alu28 = (alu26?0.0f:alu27);
        float alu29 = ((alu25!=((float)(__builtin_inff())))?alu28:0.0f);
        float alu30 = ((alu29<0.0f)?-0.5f:0.5f);
        int cast7 = ((int)((alu29+alu30)));
        int alu31 = (cast4>>1);
        int alu32 = (cast5>>1);
        int alu33 = (cast6>>1);
        int alu34 = (cast7>>1);
        float alu35 = (alu8-((float)(cast4)));
        float alu36 = (((alu4<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu35)+0.0013392627006396651f)*alu35)+0.009618384763598442f)*alu35)+0.055503472685813904f)*alu35)+0.24022644758224487f)*alu35)+0.6931471824645996f)*alu35)+1.0f)*__builtin_bit_cast(float, (int)(((alu31+127)<<23)))*__builtin_bit_cast(float, (int)((((cast4-alu31)+127)<<23)))));
        float alu37 = ((alu4<-150.0f)?0.0f:alu36);
        float alu38 = (alu5?((float)(__builtin_nanf(""))):alu37);
        float alu39 = (alu15-((float)(cast5)));
        float alu40 = (((alu11<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu39)+0.0013392627006396651f)*alu39)+0.009618384763598442f)*alu39)+0.055503472685813904f)*alu39)+0.24022644758224487f)*alu39)+0.6931471824645996f)*alu39)+1.0f)*__builtin_bit_cast(float, (int)(((alu32+127)<<23)))*__builtin_bit_cast(float, (int)((((cast5-alu32)+127)<<23)))));
        float alu41 = ((alu11<-150.0f)?0.0f:alu40);
        float alu42 = (alu12?((float)(__builtin_nanf(""))):alu41);
        float alu43 = (alu22-((float)(cast6)));
        float alu44 = (((alu18<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu43)+0.0013392627006396651f)*alu43)+0.009618384763598442f)*alu43)+0.055503472685813904f)*alu43)+0.24022644758224487f)*alu43)+0.6931471824645996f)*alu43)+1.0f)*__builtin_bit_cast(float, (int)(((alu33+127)<<23)))*__builtin_bit_cast(float, (int)((((cast6-alu33)+127)<<23)))));
        float alu45 = ((alu18<-150.0f)?0.0f:alu44);
        float alu46 = (alu19?((float)(__builtin_nanf(""))):alu45);
        float alu47 = (alu29-((float)(cast7)));
        float alu48 = (((alu25<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu47)+0.0013392627006396651f)*alu47)+0.009618384763598442f)*alu47)+0.055503472685813904f)*alu47)+0.24022644758224487f)*alu47)+0.6931471824645996f)*alu47)+1.0f)*__builtin_bit_cast(float, (int)(((alu34+127)<<23)))*__builtin_bit_cast(float, (int)((((cast7-alu34)+127)<<23)))));
        float alu49 = ((alu25<-150.0f)?0.0f:alu48);
        float alu50 = (alu26?((float)(__builtin_nanf(""))):alu49);
        *(data0_524288+(alu2+128)) = (alu3/(1.0f+alu38));
        *(data0_524288+(alu2+256)) = (alu10/(1.0f+alu42));
        *(data0_524288+(alu2+384)) = (alu17/(1.0f+alu46));
        *(data0_524288+alu2) = (alu24/(1.0f+alu50));
      }
    }
  }
}

 typedef struct {float* data0_524288;float* data1_1073152;unsigned char* data2_131072;} PoolWork;
 static void prep_callback(void **args,int unused,int group) {(void)unused;PoolWork *w=args[0];E_32_32_128_4_part(w->data0_524288,w->data1_1073152,w->data2_131072,group);}
 void E_32_32_128_4(float* restrict data0_524288, float* restrict data1_1073152, unsigned char* restrict data2_131072) {PoolWork work={data0_524288,data1_1073152,data2_131072};void *args[4]={&work,0,0,0};
  if(((int(*)(void(*)(void**,int,int),void**,int))543132856320ULL)(prep_callback,args,0))
   for(int group=0;group<4;group++)prep_callback(args,0,group);
 }
 