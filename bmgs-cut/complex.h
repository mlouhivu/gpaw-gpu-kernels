#ifndef COMPLEX_H_
#define COMPLEX_H_

typedef struct {
    double x;
    double y;
} double_complex;

double complex_abs(double_complex);

extern __host__ __device__ double_complex operator*(double_complex,
                                                    double_complex);
extern __host__ __device__ double_complex operator+(double_complex,
                                                    double_complex);
extern __host__ __device__ double_complex operator-(double_complex,
                                                    double_complex);

#endif
