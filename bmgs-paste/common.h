#ifndef COMMON_H_
#define COMMON_H_

#define BLOCK_SIZEX 32
#define BLOCK_SIZEY 16
#define BLOCK_MAX 32
#define GRID_MAX 65535
#define BLOCK_TOTALMAX 512
#define XDIV 4

#define Tcuda          double
#define Zcuda(f)       f
#define MULTT(a,b)     ((a) * (b))
#define MULTD(a,b)     ((a) * (b))
#define MULDT(a,b)     ((a) * (b))
#define ADD(a,b)       ((a) + (b))
#define ADD3(a,b,c)    ((a) + (b) + (c))
#define ADD4(a,b,c,d)  ((a) + (b) + (c) + (d))
#define IADD(a,b)      ((a) += (b))
#define MAKED(a)       (a)
#define CONJ(a)        (a)
#define REAL(a)        (a)
#define IMAG(a)        (0)
#define NEG(a)         (-(a))

#define MAX(a,b)  (((a) > (b)) ? (a) : (b))
#define MIN(a,b)  (((a) < (b)) ? (a) : (b))

static unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

typedef float (*kernel_func)(double *, const int3, const int3,
                             double *, const int3, const int,
                             char*, char*, const int, const int);

#endif
