#include "complex.h"
#include "common.h"

double complex_abs(double_complex x)
{
    double a = fabs(x.x);
    double b = fabs(x.y);
    double c;
    double k;
    if (a > b) {
        c = b / a;
        k = a;
    } else {
        c = a / b;
        k = b;
    }
    if (k == 0.0) {
        c = a + b;
    } else {
        c = k * sqrt(1.0 + c * c);
    }
    return c;
}

__host__ __device__ double_complex operator*(double_complex a,
                                             double_complex b)
{
    double_complex c;
    c.x = (a.x * b.x) - (a.y * b.y);
    c.y = (a.x * b.y) + (a.y * b.x);
    return c;
}

__host__ __device__ double_complex operator+(double_complex a,
                                             double_complex b)
{
    double_complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__host__ __device__ double_complex operator-(double_complex a,
                                             double_complex b)
{
    double_complex c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}
