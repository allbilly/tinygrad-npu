typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
typedef __fp16 __fp164 __attribute__((aligned(8),ext_vector_type(4)));
void E_32_512_4_4(float* restrict data0_262144, float* restrict data1_1048576, float* restrict data2_128, __fp16* restrict data3_2048) {
  for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    float4 val0 = (*((float4*)((data2_128+(Lidx0<<2)))));
    for (int Lidx1 = 0; Lidx1 < 512; Lidx1++) {
      int alu0 = (Lidx1<<2);
      __fp164 val1 = (*((__fp164*)((data3_2048+alu0))));
      int alu1 = ((Lidx0<<13)+alu0);
      int alu2 = (alu1+2048);
      float4 val2 = (*((float4*)((data1_1048576+alu2))));
      int alu3 = (alu1+4096);
      float4 val3 = (*((float4*)((data1_1048576+alu3))));
      int alu4 = (alu1+6144);
      float4 val4 = (*((float4*)((data1_1048576+alu4))));
      float4 val5 = (*((float4*)((data1_1048576+alu1))));
      float cast0 = ((float)(val1[0]));
      float cast1 = ((float)(val1[1]));
      float cast2 = ((float)(val1[2]));
      float cast3 = ((float)(val1[3]));
      *((float4*)((data0_262144+alu2))) = (float4){((val2[0]/val0[1])*cast0*0.00048828125f),((val2[1]/val0[1])*cast1*0.00048828125f),((val2[2]/val0[1])*cast2*0.00048828125f),((val2[3]/val0[1])*cast3*0.00048828125f)};
      *((float4*)((data0_262144+alu3))) = (float4){((val3[0]/val0[2])*cast0*0.00048828125f),((val3[1]/val0[2])*cast1*0.00048828125f),((val3[2]/val0[2])*cast2*0.00048828125f),((val3[3]/val0[2])*cast3*0.00048828125f)};
      *((float4*)((data0_262144+alu4))) = (float4){((val4[0]/val0[3])*cast0*0.00048828125f),((val4[1]/val0[3])*cast1*0.00048828125f),((val4[2]/val0[3])*cast2*0.00048828125f),((val4[3]/val0[3])*cast3*0.00048828125f)};
      *((float4*)((data0_262144+alu1))) = (float4){((val5[0]/val0[0])*cast0*0.00048828125f),((val5[1]/val0[0])*cast1*0.00048828125f),((val5[2]/val0[0])*cast2*0.00048828125f),((val5[3]/val0[0])*cast3*0.00048828125f)};
    }
  }
}
