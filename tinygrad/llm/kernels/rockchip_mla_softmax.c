/* Exact per-head softmax; canonical vector exp2 is supplied by the caller. */
static void softmax_head_rows(float *out, const float *scores, int head, int rows,
                              int start_pos, int capacity, int context, int stride) {
  int total=start_pos+rows;
  if (total>context) total=context;
  for (int row=0; row<capacity; row++) {
    int base=(head*capacity+row)*stride;
    if (row>=rows) {
      for (int k=0;k<stride;k++) out[base+k]=0.0f;
      continue;
    }
    int used=total;
    float maximum=-__builtin_inff();
    /* Preserve the original per-head sequential-key max reduction order. */
    for (int k=0;k<used;k++) if (maximum<scores[base+k]) maximum=scores[base+k];
    float sum=0.0f;
    int k=0;
    for (;k+4<=used;k+=4) {
      float lanes[4]; __builtin_memcpy(lanes,scores+base+k,sizeof(lanes));
      float4 x={lanes[0],lanes[1],lanes[2],lanes[3]};
      x=(x-(float4)maximum)*(float4)1.4426950216293335f;
      float4 e=rockchip_exp2_vector(x);
      out[base+k]=e[0]; out[base+k+1]=e[1]; out[base+k+2]=e[2]; out[base+k+3]=e[3];
      sum+=e[0]; sum+=e[1]; sum+=e[2]; sum+=e[3];
    }
    for (;k<used;k++) {
      float4 e4=rockchip_exp2_vector((float4){(scores[base+k]-maximum)*1.4426950216293335f,
        (scores[base+k]-maximum)*1.4426950216293335f,(scores[base+k]-maximum)*1.4426950216293335f,
        (scores[base+k]-maximum)*1.4426950216293335f});
      float e=e4[0];
      out[base+k]=e; sum+=e;
    }
    for (k=0;k<used;k++) out[base+k]/=sum;
    for (;k<stride;k++) out[base+k]=0.0f;
  }
}

static void softmax_head_range(float *out, const float *scores, int rows, int start_pos,
                               int capacity, int context, int stride, int first, int last) {
  for (int h=first;h<last;h++) softmax_head_rows(out,scores,h,rows,start_pos,capacity,context,stride);
}

typedef struct { float *out; const float *scores; int rows,start_pos,capacity,context,stride; } SoftmaxWork;
static void softmax_pool_callback(void **opaque,int total,int group) {
  (void)total;
  SoftmaxWork *w=(SoftmaxWork*)opaque[0];
  softmax_head_range(w->out,w->scores,w->rows,w->start_pos,w->capacity,w->context,w->stride,10*group,10*(group+1));
}
