typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
typedef __fp16 __fp164 __attribute__((aligned(8),ext_vector_type(4)));
void E_32_512_4_4(float* restrict data0_262144, float* restrict data1_1048576, float* restrict data2_262144, float* restrict data3_128, __fp16* restrict data4_2048, float* restrict data5_262144, float* restrict data6_262144, float* restrict data7_128, __fp16* restrict data8_2048) {
  for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    int alu0 = (Lidx0<<2);
    float4 val0 = (*((float4*)((data3_128+alu0))));
    float4 val1 = (*((float4*)((data7_128+alu0))));
    for (int Lidx1 = 0; Lidx1 < 512; Lidx1++) {
      int alu1 = (Lidx1<<2);
      __fp164 val2 = (*((__fp164*)((data4_2048+alu1))));
      __fp164 val3 = (*((__fp164*)((data8_2048+alu1))));
      int alu2 = ((Lidx0<<13)+alu1);
      int alu3 = (alu2+2048);
      float4 val4 = (*((float4*)((data1_1048576+alu3))));
      int alu4 = (alu2+4096);
      float4 val5 = (*((float4*)((data1_1048576+alu4))));
      int alu5 = (alu2+6144);
      float4 val6 = (*((float4*)((data1_1048576+alu5))));
      float4 val7 = (*((float4*)((data1_1048576+alu2))));
      float4 val8 = (*((float4*)((data2_262144+alu3))));
      float4 val9 = (*((float4*)((data2_262144+alu4))));
      float4 val10 = (*((float4*)((data2_262144+alu5))));
      float4 val11 = (*((float4*)((data2_262144+alu2))));
      float4 val12 = (*((float4*)((data5_262144+alu3))));
      float4 val13 = (*((float4*)((data5_262144+alu4))));
      float4 val14 = (*((float4*)((data5_262144+alu5))));
      float4 val15 = (*((float4*)((data5_262144+alu2))));
      float4 val16 = (*((float4*)((data6_262144+alu3))));
      float4 val17 = (*((float4*)((data6_262144+alu4))));
      float4 val18 = (*((float4*)((data6_262144+alu5))));
      float4 val19 = (*((float4*)((data6_262144+alu2))));
      float cast0 = ((float)(val2[0]));
      float cast1 = ((float)(val2[1]));
      float cast2 = ((float)(val2[2]));
      float cast3 = ((float)(val2[3]));
      float cast4 = ((float)(val3[0]));
      float cast5 = ((float)(val3[1]));
      float cast6 = ((float)(val3[2]));
      float cast7 = ((float)(val3[3]));
      *((float4*)((data0_262144+alu3))) = (float4){((val4[0]+((val8[0]/val0[1])*cast0)+val12[0]+((val16[0]/val1[1])*cast4))*0.7071067690849304f),((val4[1]+((val8[1]/val0[1])*cast1)+val12[1]+((val16[1]/val1[1])*cast5))*0.7071067690849304f),((val4[2]+((val8[2]/val0[1])*cast2)+val12[2]+((val16[2]/val1[1])*cast6))*0.7071067690849304f),((val4[3]+((val8[3]/val0[1])*cast3)+val12[3]+((val16[3]/val1[1])*cast7))*0.7071067690849304f)};
      *((float4*)((data0_262144+alu4))) = (float4){((val5[0]+((val9[0]/val0[2])*cast0)+val13[0]+((val17[0]/val1[2])*cast4))*0.7071067690849304f),((val5[1]+((val9[1]/val0[2])*cast1)+val13[1]+((val17[1]/val1[2])*cast5))*0.7071067690849304f),((val5[2]+((val9[2]/val0[2])*cast2)+val13[2]+((val17[2]/val1[2])*cast6))*0.7071067690849304f),((val5[3]+((val9[3]/val0[2])*cast3)+val13[3]+((val17[3]/val1[2])*cast7))*0.7071067690849304f)};
      *((float4*)((data0_262144+alu5))) = (float4){((val6[0]+((val10[0]/val0[3])*cast0)+val14[0]+((val18[0]/val1[3])*cast4))*0.7071067690849304f),((val6[1]+((val10[1]/val0[3])*cast1)+val14[1]+((val18[1]/val1[3])*cast5))*0.7071067690849304f),((val6[2]+((val10[2]/val0[3])*cast2)+val14[2]+((val18[2]/val1[3])*cast6))*0.7071067690849304f),((val6[3]+((val10[3]/val0[3])*cast3)+val14[3]+((val18[3]/val1[3])*cast7))*0.7071067690849304f)};
      *((float4*)((data0_262144+alu2))) = (float4){((val7[0]+((val11[0]/val0[0])*cast0)+val15[0]+((val19[0]/val1[0])*cast4))*0.7071067690849304f),((val7[1]+((val11[1]/val0[0])*cast1)+val15[1]+((val19[1]/val1[0])*cast5))*0.7071067690849304f),((val7[2]+((val11[2]/val0[0])*cast2)+val15[2]+((val19[2]/val1[0])*cast6))*0.7071067690849304f),((val7[3]+((val11[3]/val0[0])*cast3)+val15[3]+((val19[3]/val1[0])*cast7))*0.7071067690849304f)};
    }
  }
}
