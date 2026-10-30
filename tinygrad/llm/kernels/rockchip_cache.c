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

/* Caller proves a private, clean mapping just written by a blocking NPU command. */
int clean_from(void *memory,size_t bytes) {
  uintptr_t start=(uintptr_t)memory,end=start+bytes;
  if (line_size()!=64 || ((start|bytes)&63) || !bytes || bytes>16777216 || end<start) return -1;
  for(uintptr_t p=start;p<end;p+=64)
    __asm__ volatile("dc civac, %0" :: "r"(p) : "memory");
  __asm__ volatile("dsb sy" ::: "memory");
  return 0;
}
