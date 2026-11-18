
      typedef void (*work_fn)(void**,int,int);
      int shared_host_dispatch(work_fn fn,void **args,int position) {
        volatile int *active=(volatile int*)543146629792ULL;
        unsigned long address=*active?547627733664ULL:547627733680ULL;
        return ((int(*)(work_fn,void**,int))address)(fn,args,position);
      }
    