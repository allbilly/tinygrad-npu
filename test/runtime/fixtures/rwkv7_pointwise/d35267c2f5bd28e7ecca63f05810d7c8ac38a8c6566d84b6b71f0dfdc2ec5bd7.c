typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
typedef __fp16 __fp164 __attribute__((aligned(8),ext_vector_type(4)));
void E_32_512_4_4(float* restrict data0_262144, float* restrict data1_262144, float* restrict data2_264192, __fp16* restrict data3_2048, const int toks_) {
  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};
  for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    int alu0 = (Lidx0<<2);
    _Bool alu1 = ((alu0+1)<toks_);
    _Bool alu2 = ((alu0+2)<toks_);
    _Bool alu3 = ((alu0+3)<toks_);
    _Bool alu4 = (alu0<toks_);
    for (int Lidx1 = 0; Lidx1 < 512; Lidx1++) {
      int alu5 = (Lidx1<<2);
      __fp164 val0 = (*((__fp164*)((data3_2048+alu5))));
      int alu6 = ((Lidx0<<13)+alu5);
      int alu7 = (alu6+2048);
      float4 val1 = (alu1?*((float4*)((data1_262144+alu7))):cast0);
      int alu8 = (alu6+4096);
      float4 val2 = (alu2?*((float4*)((data1_262144+alu8))):cast0);
      int alu9 = (alu6+6144);
      float4 val3 = (alu3?*((float4*)((data1_262144+alu9))):cast0);
      float4 val4 = (alu4?*((float4*)((data1_262144+alu6))):cast0);
      float4 val5 = (alu1?*((float4*)((data2_264192+alu7))):cast0);
      float4 val6 = (alu2?*((float4*)((data2_264192+alu8))):cast0);
      float4 val7 = (alu3?*((float4*)((data2_264192+alu9))):cast0);
      float4 val8 = (alu4?*((float4*)((data2_264192+alu6))):cast0);
      float cast1 = ((float)(val0[0]));
      float cast2 = ((float)(val0[1]));
      float cast3 = ((float)(val0[2]));
      float cast4 = ((float)(val0[3]));
      float alu10 = (alu1?(val1[0]+((val5[0]-val1[0])*cast1)):0.0f);
      float alu11 = (alu1?(val1[1]+((val5[1]-val1[1])*cast2)):0.0f);
      float alu12 = (alu1?(val1[2]+((val5[2]-val1[2])*cast3)):0.0f);
      float alu13 = (alu1?(val1[3]+((val5[3]-val1[3])*cast4)):0.0f);
      float alu14 = (alu2?(val2[0]+((val6[0]-val2[0])*cast1)):0.0f);
      float alu15 = (alu2?(val2[1]+((val6[1]-val2[1])*cast2)):0.0f);
      float alu16 = (alu2?(val2[2]+((val6[2]-val2[2])*cast3)):0.0f);
      float alu17 = (alu2?(val2[3]+((val6[3]-val2[3])*cast4)):0.0f);
      float alu18 = (alu3?(val3[0]+((val7[0]-val3[0])*cast1)):0.0f);
      float alu19 = (alu3?(val3[1]+((val7[1]-val3[1])*cast2)):0.0f);
      float alu20 = (alu3?(val3[2]+((val7[2]-val3[2])*cast3)):0.0f);
      float alu21 = (alu3?(val3[3]+((val7[3]-val3[3])*cast4)):0.0f);
      float alu22 = (alu4?(val4[0]+((val8[0]-val4[0])*cast1)):0.0f);
      float alu23 = (alu4?(val4[1]+((val8[1]-val4[1])*cast2)):0.0f);
      float alu24 = (alu4?(val4[2]+((val8[2]-val4[2])*cast3)):0.0f);
      float alu25 = (alu4?(val4[3]+((val8[3]-val4[3])*cast4)):0.0f);
      *((float4*)((data0_262144+alu7))) = (float4){alu10,alu11,alu12,alu13};
      *((float4*)((data0_262144+alu8))) = (float4){alu14,alu15,alu16,alu17};
      *((float4*)((data0_262144+alu9))) = (float4){alu18,alu19,alu20,alu21};
      *((float4*)((data0_262144+alu6))) = (float4){alu22,alu23,alu24,alu25};
    }
  }
}
