typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
void E_62080_4(unsigned char* restrict args_96) {
  unsigned long val0 = (*(((unsigned long*)(args_96))+0));
  unsigned long val1 = (*(((unsigned long*)(args_96))+1));
  unsigned long val2 = (*(((unsigned long*)(args_96))+10));
  unsigned long val3 = (*(((unsigned long*)(args_96))+11));
  unsigned long val4 = (*(((unsigned long*)(args_96))+2));
  unsigned long val5 = (*(((unsigned long*)(args_96))+3));
  unsigned long val6 = (*(((unsigned long*)(args_96))+4));
  unsigned long val7 = (*(((unsigned long*)(args_96))+5));
  unsigned long val8 = (*(((unsigned long*)(args_96))+6));
  unsigned long val9 = (*(((unsigned long*)(args_96))+7));
  unsigned long val10 = (*(((unsigned long*)(args_96))+8));
  unsigned long val11 = (*(((unsigned long*)(args_96))+9));
  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};
  for (int Lidx0 = 0; Lidx0 < 62080; Lidx0++) {
    int alu0 = (Lidx0<<2);
    float4 val12 = ((Lidx0<6144)?*((float4*)((((float*)(val1))+alu0))):cast0);
    float4 val13 = (((55295<Lidx0)&(Lidx0<61440))?*((float4*)((((float*)(val2))+(alu0+-221184)))):cast0);
    float4 val14 = ((61439<Lidx0)?*((float4*)((((float*)(val3))+(alu0+-245760)))):cast0);
    float4 val15 = (((6143<Lidx0)&(Lidx0<12288))?*((float4*)((((float*)(val4))+(alu0+-24576)))):cast0);
    float4 val16 = (((12287<Lidx0)&(Lidx0<18432))?*((float4*)((((float*)(val5))+(alu0+-49152)))):cast0);
    float4 val17 = (((18431<Lidx0)&(Lidx0<24576))?*((float4*)((((float*)(val6))+(alu0+-73728)))):cast0);
    float4 val18 = (((24575<Lidx0)&(Lidx0<30720))?*((float4*)((((float*)(val7))+(alu0+-98304)))):cast0);
    float4 val19 = (((30719<Lidx0)&(Lidx0<36864))?*((float4*)((((float*)(val8))+(alu0+-122880)))):cast0);
    float4 val20 = (((36863<Lidx0)&(Lidx0<43008))?*((float4*)((((float*)(val9))+(alu0+-147456)))):cast0);
    float4 val21 = (((43007<Lidx0)&(Lidx0<49152))?*((float4*)((((float*)(val10))+(alu0+-172032)))):cast0);
    float4 val22 = (((49151<Lidx0)&(Lidx0<55296))?*((float4*)((((float*)(val11))+(alu0+-196608)))):cast0);
    *((float4*)((((float*)(val0))+alu0))) = (float4){(val12[0]+val15[0]+val16[0]+val17[0]+val18[0]+val19[0]+val20[0]+val21[0]+val22[0]+val13[0]+val14[0]),(val12[1]+val15[1]+val16[1]+val17[1]+val18[1]+val19[1]+val20[1]+val21[1]+val22[1]+val13[1]+val14[1]),(val12[2]+val15[2]+val16[2]+val17[2]+val18[2]+val19[2]+val20[2]+val21[2]+val22[2]+val13[2]+val14[2]),(val12[3]+val15[3]+val16[3]+val17[3]+val18[3]+val19[3]+val20[3]+val21[3]+val22[3]+val13[3]+val14[3])};
  }
}
