/* Declared-domain probe for the selected sinf/cosf/sin/cos; identical source for RV64 and x86-64. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__riscv)
#if __riscv_xlen != 64 || !defined(__riscv_float_abi_double)
#error RV64 lp64d required
#endif
#elif !defined(__x86_64__)
#error RV64 or x86-64 required
#endif
#if defined(__FAST_MATH__)
#error fast-math is not permitted
#endif
static float (*volatile sine32)(float)=sinf;
static float (*volatile cosine32)(float)=cosf;
static double (*volatile sine64)(double)=sin;
static double (*volatile cosine64)(double)=cos;
static unsigned char *load(const char *path,size_t unit,size_t *count) {
  FILE *in=fopen(path,"rb"); if(!in) return 0;
  size_t capacity=1<<16,size=0; unsigned char *data=malloc(capacity);
  for(;;) {
    if(!data) {fclose(in); return 0;}
    size+=fread(data+size,1,capacity-size,in);
    if(size<capacity) break;
    capacity*=2; data=realloc(data,capacity);
  }
  fclose(in);
  if(size%unit || !size) {free(data); return 0;}
  *count=size/unit; return data;
}
static int put(const void *row,size_t bytes) {return fwrite(row,1,bytes,stdout)!=bytes;}
int main(int argc,char **argv) {
  if(argc!=5) return 1;
#if defined(__riscv)
  unsigned rm; __asm__ volatile("frrm %0":"=r"(rm)); if(rm) return 2;
#else
  unsigned mxcsr; unsigned short cw;
  __asm__ volatile("stmxcsr %0":"=m"(mxcsr));
  __asm__ volatile("fnstcw %0":"=m"(cw));
  if((mxcsr & ((3u<<13)|(1u<<15)|(1u<<6))) || (cw & 0xc00u)) return 2;
#endif
  long positions=strtol(argv[1],0,10); size_t nts,nt,ns;
  unsigned char *ts=load(argv[2],4,&nts),*times=load(argv[3],4,&nt),*scaling=load(argv[4],8*360,&ns);
  if(positions<=0 || positions>(1L<<24) || !ts || nts!=32 || !times || !scaling) return 3;
  for(long p=0;p<positions;p++) for(int j=0;j<32;j++) {
    float t; memcpy(&t,ts+4*j,4);
    volatile float angle=(float)p/t;
    volatile float s=sine32(angle),c=cosine32(angle);
    float row[3]={angle,s,c};
    if(put(row,sizeof row)) return 5;
  }
  for(size_t v=0;v<ns;v++) for(size_t i=0;i<nt;i++) for(int k=0;k<360;k++) {
    float time; double scale; memcpy(&time,times+4*i,4); memcpy(&scale,scaling+8*(360*v+k),8);
    volatile double angle=scale*(double)time;
    volatile double s=sine64(angle),c=cosine64(angle);
    double row[3]={angle,s,c};
    if(put(row,sizeof row)) return 5;
  }
  return fclose(stdout)!=0;
}
