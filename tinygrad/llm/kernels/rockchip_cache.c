#include <stdint.h>
#include <stddef.h>

unsigned line_size(void) {
  uint64_t ctr;
  __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
  return 4U << ((ctr >> 16) & 15U);
}

void clean_to(void *memory,size_t bytes) {
  uintptr_t line=line_size(),start=(uintptr_t)memory&~(line-1),end=(uintptr_t)memory+bytes;
  for(uintptr_t p=start;p<end;p+=line)
    __asm__ volatile("dc cvac, %0" :: "r"(p) : "memory");
  __asm__ volatile("dsb sy" ::: "memory");
}
