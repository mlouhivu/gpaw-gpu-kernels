#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"
#include "config.h"
#include "kernel.h"


void bmgs_cut(const double *a, const int n[3], const int c[3],
              double *b, const int m[3])
{
  a += c[2] + (c[1] + c[0] * n[1]) * n[2];
  for (int i0 = 0; i0 < m[0]; i0++)
    {
      for (int i1 = 0; i1 < m[1]; i1++)
        {
          memcpy(b, a, m[2] * sizeof(double));
          a += n[2];
          b += m[2];
        }
      a += n[2] * (n[1] - m[1]);
    }
}


__global__ void Zcuda(bmgs_cut_cuda_kernel6)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = blockDim.z;
    int b = blockIdx.z;
    int t, s, tz, sz, tb, sb;
    int i, j, k;

    tb = m.z * m.y * m.x * b;
    sb = n.z * n.y * n.x * b;
    for (i = tidz; i < m.x; i += stridez) {
        tz = tb + m.z * m.y * i;
        sz = sb + n.z * n.y * i;
        for (j = tidy; j < m.y; j += stridey) {
            t = tz + m.z * j;
            s = sz + n.z * j;
            for (k = tidx; k < m.z; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}

__global__ void Zcuda(bmgs_cut_cuda_kernel5)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o, int blocks)
{
    int gridsize_y = gridDim.y / blocks;
    int b = blockIdx.y / gridsize_y;
    int tidy = threadIdx.y + (blockIdx.y - b * gridsize_y) * blockDim.y;
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidz = threadIdx.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = blockDim.z;
    int t, s, tz, sz, tb, sb;
    int i, j, k;

    tb = m.z * m.y * m.x * b;
    sb = n.z * n.y * n.x * b;
    for (i = tidz; i < m.x; i += stridez) {
        tz = tb + m.z * m.y * i;
        sz = sb + n.z * n.y * (i + o.x) + o.z;
        for (j = tidy; j < m.y; j += stridey) {
            t = tz + m.z * j;
            s = sz + n.z * (j + o.y);
            for (k = tidx; k < m.z; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}

__global__ void Zcuda(bmgs_cut_cuda_kernel4)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = blockDim.z;
    int b = blockIdx.z;
    int t, s, tz, sz, tb, sb;
    int i, j, k;

    tb = m.z * m.y * m.x * b;
    sb = n.z * n.y * n.x * b;
    for (i = tidz; i < m.x; i += stridez) {
        tz = tb + m.z * m.y * i;
        sz = sb + n.z * n.y * (i + o.x) + o.z;
        for (j = tidy; j < m.y; j += stridey) {
            t = tz + m.z * j;
            s = sz + n.z * (j + o.y);
            for (k = tidx; k < m.z; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}

__global__ void Zcuda(bmgs_cut_cuda_kernel4b)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = blockDim.z;
    int b = blockIdx.z;
    int t, s, tb, sb;
    int i, j, k;

    tb = m.z * m.y * m.x * b;
    sb = n.z * n.y * n.x * b
       + n.z * n.y * o.x
       + n.z * o.y
       + o.z;
    for (i = tidz; i < m.x; i += stridez) {
        for (j = tidy; j < m.y; j += stridey) {
            t = tb + m.z * m.y * i + m.z * j;
            s = sb + n.z * n.y * i + n.z * j;
            for (k = tidx; k < m.z; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}

__global__ void Zcuda(bmgs_cut_cuda_kernel3)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o, int blocks)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z + blockIdx.z * blockDim.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = gridDim.z * blockDim.z;
    int t, s, tz, sz, tb, sb, b;
    int i, j, k;

    for (b=0; b < blocks; b++) {
        tb = m.z * m.y * m.x * b;
        sb = n.z * n.y * n.x * b;
        for (i = tidz; i < m.x; i += stridez) {
            tz = tb + m.z * m.y * i;
            sz = sb + n.z * n.y * (i + o.x) + o.z;
            for (j = tidy; j < m.y; j += stridey) {
                t = tz + m.z * j;
                s = sz + n.z * (j + o.y);
                for (k = tidx; k < m.z; k += stridex) {
                    tgt[k + t] = src[k + s];
                }
            }
        }
    }
}

__global__ void Zcuda(bmgs_cut_cuda_kernel2)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z + blockIdx.z * blockDim.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = gridDim.z * blockDim.z;
    int t, s, tz, sz;
    int i, j, k;

    for (i = tidz; i < m.x; i += stridez) {
        tz = m.z * m.y * i;
        sz = n.z * n.y * (i + o.x) + o.z;
        for (j = tidy; j < m.y; j += stridey) {
            t = tz + m.z * j;
            s = sz + n.z * (j + o.y);
            for (k = tidx; k < m.z; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}

__global__ void Zcuda(bmgs_cut_cuda_kernel2b)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int t, s, tz, sz;
    int i;

    for (; tidy < m.x; tidy += stridey) {
        t = m.z * m.y * tidy;
        s = n.z * n.y * (tidy + o.x) + o.z;
        for (; tidx < m.y; tidx += stridex) {
            t = tz + m.z * tidx;
            s = sz + n.z * (tidy + o.y);
            for (i=0; i < m.z; i++) {
                tgt[tidx + t + i] = src[tidx + s + i];
            }
        }
    }
}


double variance(double *reference, double *result, int n)
{
    int i;
    double error = 0.0;
    double diff;

    for (i=0; i < n; i++) {
        diff = reference[i] - result[i];
        error += diff * diff;
    }
    return sqrt(error) / n;
}

void check_result(const char *name, double *y_ref, double *y, int n,
                  double time, int verbose)
{
    double error;

    error = variance(y_ref, y, n);
    if (error || verbose) {
        printf("\n%s\n", name);
        printf("reference: %f %f %f ... %f %f\n",
                y_ref[0], y_ref[1], y_ref[2], y_ref[n - 2], y_ref[n - 1]);
        printf("   result: %f %f %f ... %f %f\n",
                y[0], y[1], y[2], y[n - 2], y[n - 1]);
        printf(" variance: %f\n", error);
        printf("exec time: %f\n", time);
    }
}

void reset(double *x, double *x_, int n,
           double *y, double *y_, int m, int layers)
{
    int i;

    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);
}

int run(const unsigned int layers, const int3 sizex, const int3 sizey,
        int3 pos, float *results, char *title)
{
    int i, j, k, l, s, t;
    int verbose = 0;
    char header[512];
    char name[32];

    const int dimx[3] = {sizex.x, sizex.y, sizex.z};
    const int dimy[3] = {sizey.x, sizey.y, sizey.z};
    const int position[3] = {pos.x, pos.y, pos.z};

    const int n = dimx[0] * dimx[1] * dimx[2];
    const int m = dimy[0] * dimy[1] * dimy[2];
    double x[layers * n], y[layers * m], y_ref[layers * m];
    double *xp = &x[0];
    double *yp = &y[0];
    double *x_, *y_;
    double *xx_, *yy_;

    dim3 blocks(32, 32, 32);
    dim3 threads(16, 16, 1);
    dim3 blx, thx;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ri=0;

    // initialise data
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    // copy reference values
    for (l=0; l < layers; l++) {
        for (i=0; i < dimy[0]; i++) {
            for (j=0; j < dimy[1]; j++) {
                for (k=0; k < dimy[2]; k++) {
                    t = dimy[2] * dimy[1] * dimy[0] * l
                      + dimy[2] * dimy[1] * i
                      + dimy[2] * j
                      + k;
                    s = dimx[2] * dimx[1] * dimx[0] * l
                      + dimx[2] * dimx[1] * (i + position[0])
                      + dimx[2] * (j + position[1])
                      + k + position[2];
                    y_ref[t] = x[s];
                }
            }
        }
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n * layers);
    cudaMalloc((void **) &y_, sizeof(double) * m * layers);
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** CPU implementation ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaEventRecord(start);
    for (l=0; l < layers; l++) {
        bmgs_cut(xp + l * n, dimx, position, yp + l * m, dimy);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "CPU");
    sprintf(title, "%8s", name);
    check_result(name, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** Original GPU implementation ***/
    time = run_kernel(x_, sizex, pos, y_, sizey, layers, blx, thx,
                      title, header);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

#ifdef NONOPT_KERNELS
    /*** New GPU implementation ***/
    xx_ = x_;
    yy_ = y_;
    cudaEventRecord(start);
    for (l=0; l < layers; l++) {
        pos.x = position[0];
        pos.y = position[1];
        pos.z = position[2];
        bmgs_cut_cuda_kernel2<<<blocks, threads>>>(xx_, yy_, sizex, sizey, pos);
        xx_ += n;
        yy_ += m;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (multi-block) ***/
    cudaEventRecord(start);
    bmgs_cut_cuda_kernel3<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL3");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (multi-block, block in dim) ***/
    blx = {32, 32, layers};
    cudaEventRecord(start);
    bmgs_cut_cuda_kernel4<<<blx, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL4");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blx.x, blx.y, blx.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (multi-block, block in dim) ***/
    blx = {32, 32 * layers, 1};
    cudaEventRecord(start);
    bmgs_cut_cuda_kernel5<<<blx, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL5");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blx.x, blx.y, blx.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (multi-block, block in dim) ***/
    blx = {32, 32, layers};
    xx_ = x_;
    cudaEventRecord(start);
    xx_ += dimx[2] * dimx[1] * position[0]
         + dimx[2] * position[1]
         + position[2];
    bmgs_cut_cuda_kernel6<<<blx, threads>>>(
            xx_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL6");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blx.x, blx.y, blx.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;
#endif

    /*** OPTIMISED KERNELS ***/

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised) ***/
    xx_ = x_;
    yy_ = y_;
    cudaEventRecord(start);
    threads.x = min(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = min(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = (dimy[0] + threads.z - 1) / threads.z;
    for (l=0; l < layers; l++) {
        pos.x = position[0];
        pos.y = position[1];
        pos.z = position[2];
        bmgs_cut_cuda_kernel2<<<blocks, threads>>>(xx_, yy_, sizex, sizey, pos);
        xx_ += n;
        yy_ += m;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised multi-block) ***/
    cudaEventRecord(start);
    threads.x = min(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = min(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = (dimy[0] + threads.z - 1) / threads.z;
    bmgs_cut_cuda_kernel3<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL3");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised multi-block) ***/
    cudaEventRecord(start);
    threads.x = min(nextPow2(dimy[2]), BLOCK_MAX);
    threads.y = min(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = (dimy[0] + threads.z - 1) / threads.z;
    bmgs_cut_cuda_kernel3<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERN3v2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = layers;
    bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL4");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_MAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = layers;
    bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERN4v2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_MAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_MAX / threads.x);
    threads.z = BLOCK_MAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = layers;
    bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERN4v3");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

#ifdef DEFUNC
    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = layers * (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = 1;
    bmgs_cut_cuda_kernel5<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL5");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = 1;
    threads.y = MIN(MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x),
                    BLOCK_MAX);
    threads.z = BLOCK_MAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = layers * (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = 1;
    bmgs_cut_cuda_kernel5<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL5v2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

    /*** reset ***/
    reset(x, x_, n, y, y_, m, layers);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    xx_ = x_;
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = layers;
    xx_ += dimx[2] * dimx[1] * position[0]
         + dimx[2] * position[1]
         + position[2];
    bmgs_cut_cuda_kernel6<<<blocks, threads>>>(
            xx_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL6");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;
#endif

    return ri;
}

int main(void)
{
    int i, j;

    int ri = 0;
    int rj = 0;
    int trials = 512;
    int kernels = 512;
    float results[trials][kernels];
    float best[trials];
    float total[kernels];
    float time;

    char title[512];

    int layers;
    int3 dimx;
    int3 dimy;
    int3 position;
    t_config config;
    t_arg arg;

    for (i=0; i < kernels; i++)
        total[i] = 0.0;

    config = get_config();
    for (i=0; i < config.nargs; i++) {
        arg = config.args[i];
        j = run(arg.layers, arg.dimx, arg.dimy, arg.position,
                results[ri++], title);
        rj = MAX(rj, j);
    }

    printf("\nTiming results:\n%s\n", title);
    for (i=0; i < ri; i++) {
        best[i] = 9999.0;
        for (j=0; j < rj; j++) {
            printf("%f ", results[i][j]);
            best[i] = MIN(best[i], results[i][j]);
        }
        printf("\n");
    }
    printf("\nCompared to the best:\n%s\n", title);
    for (i=0; i < ri; i++) {
        for (j=0; j < rj; j++) {
            time = results[i][j] - best[i];
            if (time > 0.0)
                printf("%f ", time);
            else
                printf("-------- ");
            total[j] += time;
        }
        printf("\n");
    }
    printf("\nTotal (lost) time:\n%s\n", title);
    for (j=0; j < rj; j++) {
        printf("%f ", total[j]);
    }
    printf("\n");

    return 0;
}
