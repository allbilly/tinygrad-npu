typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
typedef __fp16 __fp164 __attribute__((aligned(8),ext_vector_type(4)));
void E_32_512_4_4(float* restrict data0_262144, float* restrict data1_262144, float* restrict data2_262144, float* restrict data3_262144, float* restrict data4_128, __fp16* restrict data5_2048) {
  for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    float4 val0 = (*((float4*)((data4_128+(Lidx0<<2)))));
    for (int Lidx1 = 0; Lidx1 < 512; Lidx1++) {
      int alu0 = (Lidx1<<2);
      __fp164 val1 = (*((__fp164*)((data5_2048+alu0))));
      int alu1 = ((Lidx0<<13)+alu0);
      int alu2 = (alu1+2048);
      float4 val2 = (*((float4*)((data1_262144+alu2))));
      int alu3 = (alu1+4096);
      float4 val3 = (*((float4*)((data1_262144+alu3))));
      int alu4 = (alu1+6144);
      float4 val4 = (*((float4*)((data1_262144+alu4))));
      float4 val5 = (*((float4*)((data1_262144+alu1))));
      float4 val6 = (*((float4*)((data2_262144+alu2))));
      float4 val7 = (*((float4*)((data2_262144+alu3))));
      float4 val8 = (*((float4*)((data2_262144+alu4))));
      float4 val9 = (*((float4*)((data2_262144+alu1))));
      float4 val10 = (*((float4*)((data3_262144+alu2))));
      float4 val11 = (*((float4*)((data3_262144+alu3))));
      float4 val12 = (*((float4*)((data3_262144+alu4))));
      float4 val13 = (*((float4*)((data3_262144+alu1))));
      float cast0 = ((float)(val1[0]));
      float cast1 = ((float)(val1[1]));
      float cast2 = ((float)(val1[2]));
      float cast3 = ((float)(val1[3]));
      *((float4*)((data0_262144+alu2))) = (float4){(val2[0]+(((val6[0]+val10[0])/val0[1])*cast0)),(val2[1]+(((val6[1]+val10[1])/val0[1])*cast1)),(val2[2]+(((val6[2]+val10[2])/val0[1])*cast2)),(val2[3]+(((val6[3]+val10[3])/val0[1])*cast3))};
      *((float4*)((data0_262144+alu3))) = (float4){(val3[0]+(((val7[0]+val11[0])/val0[2])*cast0)),(val3[1]+(((val7[1]+val11[1])/val0[2])*cast1)),(val3[2]+(((val7[2]+val11[2])/val0[2])*cast2)),(val3[3]+(((val7[3]+val11[3])/val0[2])*cast3))};
      *((float4*)((data0_262144+alu4))) = (float4){(val4[0]+(((val8[0]+val12[0])/val0[3])*cast0)),(val4[1]+(((val8[1]+val12[1])/val0[3])*cast1)),(val4[2]+(((val8[2]+val12[2])/val0[3])*cast2)),(val4[3]+(((val8[3]+val12[3])/val0[3])*cast3))};
      *((float4*)((data0_262144+alu1))) = (float4){(val5[0]+(((val9[0]+val13[0])/val0[0])*cast0)),(val5[1]+(((val9[1]+val13[1])/val0[0])*cast1)),(val5[2]+(((val9[2]+val13[2])/val0[0])*cast2)),(val5[3]+(((val9[3]+val13[3])/val0[0])*cast3))};
    }
  }
}
