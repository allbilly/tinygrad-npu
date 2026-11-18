typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef int int4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));static __attribute__((always_inline)) float4 rockchip_exp2_slow(float4 x) {int4 nan=x!=x;
    float4 safe=((x==(float4)__builtin_inff())|(x==(float4)-__builtin_inff())|nan)?(float4)0:x;
    int4 q=__builtin_convertvector(safe+(safe<0?(float4)-.5f:(float4).5f),int4),q1=q>>1;
    float4 s=safe-__builtin_convertvector(q,float4);
    float4 y=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
      0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f)*__builtin_bit_cast(float4,(q1+127)<<23)*__builtin_bit_cast(float4,(q-q1+127)<<23);
    y=x>=128?(float4)__builtin_inff():y;
    y=x<-150?(float4)0:y;
    y=nan?(float4)__builtin_nanf(""):y;
    return y;}
  static __attribute__((always_inline)) float4 rockchip_exp2_vector(float4 x) {
    int4 magnitude=__builtin_bit_cast(int4,x)&(int4)0x7fffffff;
    if(__builtin_expect(__builtin_reduce_max(magnitude)>=0x42fa0000,0))return rockchip_exp2_slow(x);
    int4 q=__builtin_convertvector(x+(x<0?(float4)-.5f:(float4).5f),int4);
    float4 s=x-__builtin_convertvector(q,float4);
    float4 y=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
      0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f);
    return __builtin_bit_cast(float4,__builtin_bit_cast(int4,y)+q*(int4)0x800000);
  }typedef signed char char4 __attribute__((aligned(1),ext_vector_type(4)));
  static __attribute__((always_inline)) float4 rockchip_round_even(float4 x) {
    #if defined(__aarch64__)
    float4 result;__asm__("frintn %0.4s, %1.4s":"=w"(result):"w"(x));return result;
    #else
    return (float4){__builtin_nearbyintf(x[0]),__builtin_nearbyintf(x[1]),__builtin_nearbyintf(x[2]),__builtin_nearbyintf(x[3])};
    #endif
  }
  static __attribute__((always_inline)) char4 rockchip_quantize_vector(float4 x,float inverse) {
    float4 q=rockchip_round_even(x*(float4)inverse);
    q=q>(float4)127?(float4)127:q;q=q<(float4)-127?(float4)-127:q;
    return __builtin_convertvector(q,char4);
  }
  typedef unsigned uint4 __attribute__((aligned(16),ext_vector_type(4)));
  static __attribute__((always_inline)) float4 project(const int *const *raw,const float *const *sa,const float *sw,int row,int column) {
    if(row>=0 && row<64 && column>=0 && column<4096) {
        int position=0+(row-0)*4096+column-0;
        float4 value=__builtin_convertvector(*(const int4*)(raw[0]+position),float4);
        return (value*(float4)sa[0][row-0])*(*(const float4*)(sw+column));
      }if(row>=0 && row<64 && column>=4096 && column<8192) {
        int position=262144+(row-0)*4096+column-4096;
        float4 value=__builtin_convertvector(*(const int4*)(raw[0]+position),float4);
        return (value*(float4)sa[0][row-0])*(*(const float4*)(sw+column));
      }if(row>=0 && row<64 && column>=8192 && column<12288) {
        int position=524288+(row-0)*4096+column-8192;
        float4 value=__builtin_convertvector(*(const int4*)(raw[0]+position),float4);
        return (value*(float4)sa[0][row-0])*(*(const float4*)(sw+column));
      }if(row>=0 && row<64 && column>=12288 && column<14336) {
        int position=0+(row-0)*2048+column-12288;
        float4 value=__builtin_convertvector(*(const int4*)(raw[1]+position),float4);
        return (value*(float4)sa[0][row-0])*(*(const float4*)(sw+column));
      }if(row>=0 && row<64 && column>=14336 && column<16384) {
        int position=131072+(row-0)*2048+column-14336;
        float4 value=__builtin_convertvector(*(const int4*)(raw[1]+position),float4);
        return (value*(float4)sa[0][row-0])*(*(const float4*)(sw+column));
      }if(row>=0 && row<64 && column>=16384 && column<18432) {
        int position=262144+(row-0)*2048+column-16384;
        float4 value=__builtin_convertvector(*(const int4*)(raw[1]+position),float4);
        return (value*(float4)sa[0][row-0])*(*(const float4*)(sw+column));
      }if(row>=64 && row<128 && column>=0 && column<4096) {
        int position=0+(row-64)*4096+column-0;
        float4 value=__builtin_convertvector(*(const int4*)(raw[2]+position),float4);
        return (value*(float4)sa[1][row-64])*(*(const float4*)(sw+column));
      }if(row>=64 && row<128 && column>=4096 && column<8192) {
        int position=262144+(row-64)*4096+column-4096;
        float4 value=__builtin_convertvector(*(const int4*)(raw[2]+position),float4);
        return (value*(float4)sa[1][row-64])*(*(const float4*)(sw+column));
      }if(row>=64 && row<128 && column>=8192 && column<12288) {
        int position=524288+(row-64)*4096+column-8192;
        float4 value=__builtin_convertvector(*(const int4*)(raw[2]+position),float4);
        return (value*(float4)sa[1][row-64])*(*(const float4*)(sw+column));
      }if(row>=64 && row<128 && column>=12288 && column<14336) {
        int position=0+(row-64)*2048+column-12288;
        float4 value=__builtin_convertvector(*(const int4*)(raw[3]+position),float4);
        return (value*(float4)sa[1][row-64])*(*(const float4*)(sw+column));
      }if(row>=64 && row<128 && column>=14336 && column<16384) {
        int position=131072+(row-64)*2048+column-14336;
        float4 value=__builtin_convertvector(*(const int4*)(raw[3]+position),float4);
        return (value*(float4)sa[1][row-64])*(*(const float4*)(sw+column));
      }if(row>=64 && row<128 && column>=16384 && column<18432) {
        int position=262144+(row-64)*2048+column-16384;
        float4 value=__builtin_convertvector(*(const int4*)(raw[3]+position),float4);
        return (value*(float4)sa[1][row-64])*(*(const float4*)(sw+column));
      }return (float4)__builtin_nanf("");
  }
  static void part_g0(unsigned char *out,const int *const *raw,const float *const *sa,const float *sw,float *scales,int active,int begin,int end) {
    for(int row=begin;row<end;row++) {
      if(row+4<=end && row+4<=active && ((row>=0 && row+4<=64) || (row>=64 && row+4<=128))) {
    int batch_start=row;float batch_values[4][4096] __attribute__((aligned(16)));
    float4 batch_maximum[4]={0};uint4 batch_bad[4]={0};
    if(row>=0 && row+4<=64) {for(int c=0;c<3072;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+((row+0)-0)*4096+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[0]+(525312+((row+0)-0)*4096+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+((row+1)-0)*4096+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[0]+(525312+((row+1)-0)*4096+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+((row+2)-0)*4096+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[0]+(525312+((row+2)-0)*4096+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+((row+3)-0)*4096+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[0]+(525312+((row+3)-0)*4096+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}for(int c=3072;c<4096;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+((row+0)-0)*4096+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(-3072+((row+0)-0)*2048+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+((row+1)-0)*4096+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(-3072+((row+1)-0)*2048+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+((row+2)-0)*4096+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(-3072+((row+2)-0)*2048+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+((row+3)-0)*4096+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(-3072+((row+3)-0)*2048+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}}if(row>=64 && row+4<=128) {for(int c=0;c<3072;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+((row+0)-64)*4096+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[2]+(525312+((row+0)-64)*4096+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+((row+1)-64)*4096+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[2]+(525312+((row+1)-64)*4096+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+((row+2)-64)*4096+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[2]+(525312+((row+2)-64)*4096+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+((row+3)-64)*4096+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[2]+(525312+((row+3)-64)*4096+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}for(int c=3072;c<4096;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+((row+0)-64)*4096+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(-3072+((row+0)-64)*2048+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+((row+1)-64)*4096+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(-3072+((row+1)-64)*2048+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+((row+2)-64)*4096+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(-3072+((row+2)-64)*2048+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+((row+3)-64)*4096+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(-3072+((row+3)-64)*2048+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+9216+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}}
    {int row=batch_start+0;float *values=batch_values[0];
      float4 maximum=batch_maximum[0];uint4 bad=batch_bad[0];
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+0+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[0+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }{int row=batch_start+1;float *values=batch_values[1];
      float4 maximum=batch_maximum[1];uint4 bad=batch_bad[1];
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+0+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[0+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }{int row=batch_start+2;float *values=batch_values[2];
      float4 maximum=batch_maximum[2];uint4 bad=batch_bad[2];
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+0+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[0+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }{int row=batch_start+3;float *values=batch_values[3];
      float4 maximum=batch_maximum[3];uint4 bad=batch_bad[3];
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+0+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[0+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }row+=3;continue;
  }
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+0+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      if(row>=active) {((float*)base)[row-first]=scales[0+row]=1e-8f;
        for(int c=0;c<4096;c++)dest[c]=0;continue;}
      float values[4096] __attribute__((aligned(16)));float4 maximum={0};uint4 bad={0};
      if(row>=0 && row<64) {for(int c=0;c<3072;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+(row-0)*4096+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[0]+(525312+(row-0)*4096+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+9216+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }for(int c=3072;c<4096;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(0+(row-0)*4096+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(-3072+(row-0)*2048+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+9216+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}if(row>=64 && row<128) {for(int c=0;c<3072;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+(row-64)*4096+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[2]+(525312+(row-64)*4096+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+9216+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }for(int c=3072;c<4096;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(0+(row-64)*4096+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+0+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(-3072+(row-64)*2048+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+9216+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[0+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }
  } static void part_g1(unsigned char *out,const int *const *raw,const float *const *sa,const float *sw,float *scales,int active,int begin,int end) {
    for(int row=begin;row<end;row++) {
      if(row+4<=end && row+4<=active && ((row>=0 && row+4<=64) || (row>=64 && row+4<=128))) {
    int batch_start=row;float batch_values[4][4096] __attribute__((aligned(16)));
    float4 batch_maximum[4]={0};uint4 batch_bad[4]={0};
    if(row>=0 && row+4<=64) {for(int c=0;c<1024;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+0)-0)*4096+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(1024+((row+0)-0)*2048+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+1)-0)*4096+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(1024+((row+1)-0)*2048+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+2)-0)*4096+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(1024+((row+2)-0)*2048+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+3)-0)*4096+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(1024+((row+3)-0)*2048+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}for(int c=1024;c<3072;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+0)-0)*4096+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(130048+((row+0)-0)*2048+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+1)-0)*4096+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(130048+((row+1)-0)*2048+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+2)-0)*4096+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(130048+((row+2)-0)*2048+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+3)-0)*4096+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(130048+((row+3)-0)*2048+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}for(int c=3072;c<4096;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+0)-0)*4096+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(259072+((row+0)-0)*2048+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+1)-0)*4096+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(259072+((row+1)-0)*2048+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+2)-0)*4096+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(259072+((row+2)-0)*2048+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+((row+3)-0)*4096+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(259072+((row+3)-0)*2048+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}}if(row>=64 && row+4<=128) {for(int c=0;c<1024;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+0)-64)*4096+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(1024+((row+0)-64)*2048+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+1)-64)*4096+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(1024+((row+1)-64)*2048+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+2)-64)*4096+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(1024+((row+2)-64)*2048+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+3)-64)*4096+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(1024+((row+3)-64)*2048+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}for(int c=1024;c<3072;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+0)-64)*4096+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(130048+((row+0)-64)*2048+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+1)-64)*4096+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(130048+((row+1)-64)*2048+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+2)-64)*4096+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(130048+((row+2)-64)*2048+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+3)-64)*4096+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(130048+((row+3)-64)*2048+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}for(int c=3072;c<4096;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+0)-64)*4096+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(259072+((row+0)-64)*2048+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+1)-64)*4096+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(259072+((row+1)-64)*2048+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+2)-64)*4096+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(259072+((row+2)-64)*2048+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+((row+3)-64)*4096+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(259072+((row+3)-64)*2048+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+13312+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}}
    {int row=batch_start+0;float *values=batch_values[0];
      float4 maximum=batch_maximum[0];uint4 bad=batch_bad[0];
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+524800+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[128+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }{int row=batch_start+1;float *values=batch_values[1];
      float4 maximum=batch_maximum[1];uint4 bad=batch_bad[1];
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+524800+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[128+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }{int row=batch_start+2;float *values=batch_values[2];
      float4 maximum=batch_maximum[2];uint4 bad=batch_bad[2];
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+524800+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[128+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }{int row=batch_start+3;float *values=batch_values[3];
      float4 maximum=batch_maximum[3];uint4 bad=batch_bad[3];
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+524800+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[128+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }row+=3;continue;
  }
      int first=row/64*64,count=128-first<64?128-first:64;
      unsigned char *base=out+524800+first*4096+first/64*256;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*4096;
      if(row>=active) {((float*)base)[row-first]=scales[128+row]=1e-8f;
        for(int c=0;c<4096;c++)dest[c]=0;continue;}
      float values[4096] __attribute__((aligned(16)));float4 maximum={0};uint4 bad={0};
      if(row>=0 && row<64) {for(int c=0;c<1024;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+(row-0)*4096+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(1024+(row-0)*2048+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+13312+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }for(int c=1024;c<3072;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+(row-0)*4096+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(130048+(row-0)*2048+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+13312+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }for(int c=3072;c<4096;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(262144+(row-0)*4096+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(259072+(row-0)*2048+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+13312+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}if(row>=64 && row<128) {for(int c=0;c<1024;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+(row-64)*4096+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(1024+(row-64)*2048+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+13312+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }for(int c=1024;c<3072;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+(row-64)*4096+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(130048+(row-64)*2048+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+13312+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }for(int c=3072;c<4096;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(262144+(row-64)*4096+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+4096+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(259072+(row-64)*2048+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+13312+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[128+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<4096;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<4096;c++)dest[c]=0;
    }
  } static void part_g2(unsigned char *out,const int *const *raw,const float *const *sa,const float *sw,float *scales,int active,int begin,int end) {
    for(int row=begin;row<end;row++) {
      if(row+4<=end && row+4<=active && ((row>=0 && row+4<=64) || (row>=64 && row+4<=128))) {
    int batch_start=row;float batch_values[4][1024] __attribute__((aligned(16)));
    float4 batch_maximum[4]={0};uint4 batch_bad[4]={0};
    if(row>=0 && row+4<=64) {for(int c=0;c<1024;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(524288+((row+0)-0)*4096+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(263168+((row+0)-0)*2048+c)),float4)*
            (float4)sa[0][(row+0)-0])*(*(const float4*)(sw+17408+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(524288+((row+1)-0)*4096+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(263168+((row+1)-0)*2048+c)),float4)*
            (float4)sa[0][(row+1)-0])*(*(const float4*)(sw+17408+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(524288+((row+2)-0)*4096+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(263168+((row+2)-0)*2048+c)),float4)*
            (float4)sa[0][(row+2)-0])*(*(const float4*)(sw+17408+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(524288+((row+3)-0)*4096+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(263168+((row+3)-0)*2048+c)),float4)*
            (float4)sa[0][(row+3)-0])*(*(const float4*)(sw+17408+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}}if(row>=64 && row+4<=128) {for(int c=0;c<1024;c+=4) {{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(524288+((row+0)-64)*4096+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(263168+((row+0)-64)*2048+c)),float4)*
            (float4)sa[1][(row+0)-64])*(*(const float4*)(sw+17408+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[0]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[0]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[0]=a>batch_maximum[0]?a:batch_maximum[0];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(524288+((row+1)-64)*4096+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(263168+((row+1)-64)*2048+c)),float4)*
            (float4)sa[1][(row+1)-64])*(*(const float4*)(sw+17408+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[1]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[1]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[1]=a>batch_maximum[1]?a:batch_maximum[1];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(524288+((row+2)-64)*4096+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(263168+((row+2)-64)*2048+c)),float4)*
            (float4)sa[1][(row+2)-64])*(*(const float4*)(sw+17408+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[2]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[2]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[2]=a>batch_maximum[2]?a:batch_maximum[2];}{float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(524288+((row+3)-64)*4096+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(263168+((row+3)-64)*2048+c)),float4)*
            (float4)sa[1][(row+3)-64])*(*(const float4*)(sw+17408+c));float4 exponent=g*-1.4426950216293335f;float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(batch_values[3]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[3]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[3]=a>batch_maximum[3]?a:batch_maximum[3];}}}
    {int row=batch_start+0;float *values=batch_values[0];
      float4 maximum=batch_maximum[0];uint4 bad=batch_bad[0];
      int first=row/128*128,count=128-first<128?128-first:128;
      unsigned char *base=out+1049600+first*1024+first/128*512;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*1024;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[256+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<1024;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<1024;c++)dest[c]=0;
    }{int row=batch_start+1;float *values=batch_values[1];
      float4 maximum=batch_maximum[1];uint4 bad=batch_bad[1];
      int first=row/128*128,count=128-first<128?128-first:128;
      unsigned char *base=out+1049600+first*1024+first/128*512;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*1024;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[256+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<1024;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<1024;c++)dest[c]=0;
    }{int row=batch_start+2;float *values=batch_values[2];
      float4 maximum=batch_maximum[2];uint4 bad=batch_bad[2];
      int first=row/128*128,count=128-first<128?128-first:128;
      unsigned char *base=out+1049600+first*1024+first/128*512;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*1024;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[256+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<1024;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<1024;c++)dest[c]=0;
    }{int row=batch_start+3;float *values=batch_values[3];
      float4 maximum=batch_maximum[3];uint4 bad=batch_bad[3];
      int first=row/128*128,count=128-first<128?128-first:128;
      unsigned char *base=out+1049600+first*1024+first/128*512;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*1024;
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[256+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<1024;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<1024;c++)dest[c]=0;
    }row+=3;continue;
  }
      int first=row/128*128,count=128-first<128?128-first:128;
      unsigned char *base=out+1049600+first*1024+first/128*512;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*1024;
      if(row>=active) {((float*)base)[row-first]=scales[256+row]=1e-8f;
        for(int c=0;c<1024;c++)dest[c]=0;continue;}
      float values[1024] __attribute__((aligned(16)));float4 maximum={0};uint4 bad={0};
      if(row>=0 && row<64) {for(int c=0;c<1024;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[0]+(524288+(row-0)*4096+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[1]+(263168+(row-0)*2048+c)),float4)*
          (float4)sa[0][row-0])*(*(const float4*)(sw+17408+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}if(row>=64 && row<128) {for(int c=0;c<1024;c+=4) {
        float4 g=(__builtin_convertvector(*(const int4*)(raw[2]+(524288+(row-64)*4096+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+8192+c)),u=(__builtin_convertvector(*(const int4*)(raw[3]+(263168+(row-64)*2048+c)),float4)*
          (float4)sa[1][row-64])*(*(const float4*)(sw+17408+c));
        float4 exponent=g*-1.4426950216293335f;
        float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[256+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<1024;c+=4){float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}
      for(;c<1024;c++)dest[c]=0;
    }
  }
  static void part(unsigned char *out,const int *const *raw,const float *const *sa,const float *sw,float *scales,int active,int begin,int end) {
    part_g0(out,raw,sa,sw,scales,active,begin,end); part_g1(out,raw,sa,sw,scales,active,begin,end); part_g2(out,raw,sa,sw,scales,active,begin,end);
  }
  typedef struct {unsigned char *out;const int *const *raw;const float *const *sa;
      const float *sw;float *scales;int active;} Work;
    static void pool_part(void **args,int unused,int group){
      Work *w=args[0];part(w->out,w->raw,w->sa,w->sw,w->scales,w->active,
                          128*group/4,128*(group+1)/4);
    }
  void rockchip_int8_swiglu_pack(unsigned char *out,const int *r0,const int *r1,const int *r2,const int *r3,const float *a0,const float *a1,const float *sw,const int *active,float *scales) {
    if(*active<1 || *active>128)__builtin_trap();const int *raw[4]={r0,r1,r2,r3};
    const float *sa[2]={a0,a1};for(int i=256;i<256;i++)out[0+i]=0;for(int i=256;i<256;i++)out[262400+i]=0;for(int i=256;i<256;i++)out[524800+i]=0;for(int i=256;i<256;i++)out[787200+i]=0;for(int i=512;i<512;i++)out[1049600+i]=0;Work work={out,raw,sa,sw,scales,*active};void *args[4]={&work,0,0,0};
      if(((int(*)(void(*)(void**,int,int),void**,int))543132856320ULL)(pool_part,args,0))__builtin_trap();
  }