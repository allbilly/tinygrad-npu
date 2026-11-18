static void shared_work_callback(void**,int,int);
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  static void norm_part(float*,const float*,const __fp16*,int,int);
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {float *out;const float *src;const __fp16 *weight;int begin,end;} Work;
  static void *norm_worker(void*);
  void rockchip_cpu_rmsnorm(float *out,const float *src,const __fp16 *weight,const int *active_rows) {
    Work work[4];
    for(int i=0;i<4;i++)work[i]=(Work){out,src,weight,128*i/4,128*(i+1)/4};
    void *shared_args[4]={work,0,0,0};
    if(((int(*)(void(*)(void**,int,int),void**,int))543132856320ULL)(shared_work_callback,shared_args,0)){
      for(int i=0;i<4;i++)norm_worker(work+i);
    }if(*active_rows>=0 && *active_rows<128)__builtin_memset(out+*active_rows*2560,0,(128-*active_rows)*2560UL*4);
  }
  static void norm_part(float *out,const float *src,const __fp16 *weight,int begin,int end) {
    for(int r=begin;r<end;r++) {
      float4 acc=(float4)0;
      for(int c=0;c<2560;c+=4) {float4 value=*(const float4*)(src+r*2560+c);acc=acc+value*value;}
      float sum=((acc[0]+acc[1])+acc[2])+acc[3];
      float inverse=1.0f/__builtin_sqrtf(sum*0.00039062500000000002f+9.9999999747524271e-07f);
      for(int c=0;c<2560;c+=4) {float4 value=*(const float4*)(src+r*2560+c),gamma=__builtin_convertvector(*(const half4*)(weight+c),float4);
        *(float4*)(out+r*2560+c)=(value*(float4)inverse)*gamma;}
    }
  }
  static void *norm_worker(void *opaque) {
    Work *w=(Work*)opaque;norm_part(w->out,w->src,w->weight,w->begin,w->end);return (void*)0;
  }
    static void shared_work_callback(void **args,int unused,int group) {
      (void)unused;norm_worker(((Work*)args[0])+group);
    }
  