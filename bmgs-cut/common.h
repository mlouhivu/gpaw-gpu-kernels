#ifndef COMMON_H_
#define COMMON_H_

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "complex.h"

#define BLOCK_MAX 32
#define GRID_MAX 65535
#define BLOCK_TOTALMAX 256

#ifndef USE_COMPLEX

#define T              double
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

#define ABS(a)         (fabs(a))

#else

#define T              double_complex
#define Tcuda          cuDoubleComplex
#define Zcuda(f)       f ## z
#define MULTT(a,b)     cuCmul((a), (b))
#define MULTD(a,b)     cuCmulD((a), (b))
#define MULDT(b,a)     MULTD((a), (b))
#define ADD(a,b)       cuCadd((a), (b))
#define ADD3(a,b,c)    cuCadd3((a), (b), (c))
#define ADD4(a,b,c,d)  cuCadd4((a), (b), (c), (d))
#define IADD(a,b)      {(a).x += cuCreal(b); (a).y += cuCimag(b);}
#define MAKED(a)       make_cuDoubleComplex(a, 0)
#define CONJ(a)        cuConj(a)
#define REAL(a)        cuCreal(a)
#define IMAG(a)        cuCimag(a)
#define NEG(a)         cuCneg(a)

#define ABS(a)         (complex_abs(a))

#endif

#define AS_T(r,i)      as_T(r,i)
#define AS_TCUDA(r,i)  as_Tcuda(r,i)

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

typedef float (*kernel_func)(Tcuda *, const int3, const int3,
                             Tcuda *, const int3, const int, const Tcuda,
                             char*, char*, const int, const int);

__host__ __device__ static __inline__ cuDoubleComplex cuCneg(
        cuDoubleComplex x)
{
    return make_cuDoubleComplex(-cuCreal(x), -cuCimag(x));
}

__host__ __device__ static __inline__ T as_T(double x, double y)
{
    T result;
#ifdef USE_COMPLEX
    result.x = x;
    result.y = y;
#else
    result = x;
#endif
    return result;
}

__host__ __device__ static __inline__ Tcuda as_Tcuda(double x, double y)
{
#ifdef USE_COMPLEX
    return make_cuDoubleComplex(x, y);
#else
    return (Tcuda) x;
#endif
}

#endif
