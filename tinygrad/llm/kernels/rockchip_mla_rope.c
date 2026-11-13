#include <stddef.h>
/* Fused MLA query-head kernel: split the q_b projection output into the padded
   head-major q_nope and the RoPE-rotated q_rope in one pass, so neither the
   transpose copy nor the q_nope pad nor the separate rope kernel is needed.
   The rotation reproduces the reference (x1*cos - x2*sin, x2*cos + x1*sin)
   element order exactly. */
static void mla_query_head_part(float *nope,float *rope,const float *src,const float *table,
                                int heads,int capacity,int rows,int start,int row_begin,int row_end) {
  const int row_width=heads*96;
  for(int r=row_begin;r<row_end;r++) {
    const float *s=src+(size_t)r*row_width;
    const float *cs=table+(size_t)(start+r)*32;
    for(int h=0;h<heads;h++) {
      const float *q=s+(size_t)h*96;
      float *n=nope+((size_t)h*capacity+r)*64;
      float *ro=rope+((size_t)h*capacity+r)*32;
      for(int d=0;d<64;d++)n[d]=q[d];
      for(int d=0;d<16;d++) {
        float x1=q[64+d],x2=q[80+d];
        ro[d]=(x1*cs[d])-(x2*cs[16+d]);
        ro[16+d]=(x2*cs[d])+(x1*cs[16+d]);
      }
    }
  }
}
void rockchip_mla_query_head(float *nope,float *rope,const float *src,const float *table,const int *start,const int *rows) {
  if(*start<0||*rows<1||*rows>128||*start+*rows>1024)__builtin_trap();
  for(int i=0;i<40*128*64;i++)nope[i]=0.0f;
  for(int i=0;i<40*128*32;i++)rope[i]=0.0f;
  mla_query_head_part(nope,rope,src,table,40,128,*rows,*start,0,*rows);
}
