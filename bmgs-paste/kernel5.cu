#include "kernels.h"

__global__ void Zcuda(bmgs_cut_cuda_kernel5)(
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
    for (i = tidz; i < n.x; i += stridez) {
        tz = tb + m.z * m.y * i;
        sz = sb + n.z * n.y * i;
        for (j = tidy; j < n.y; j += stridey) {
            t = tz + m.z * j;
            s = sz + n.z * j;
            for (k = tidx; k < n.z; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}

/*** New GPU implementation (multi-block, block in dim) ***/
float run_kernel5(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey, const int layers,
                  char *title, char *header,
                  const int repeat, const int trial)
{
    const int dimx[3] = {sizex.x, sizex.y, sizex.z};
    const int dimy[3] = {sizey.x, sizey.y, sizey.z};
    const int position[3] = {pos.x, pos.y, pos.z};

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blocks, threads;

    double *yy_;

    char name[32];

    cudaEventRecord(start);
    for (int i=0; i < repeat; i++) {
        yy_ = y_;
        threads.x = MIN(nextPow2(dimx[2]), BLOCK_TOTALMAX);
        threads.y = MIN(nextPow2(dimx[1]), BLOCK_TOTALMAX / threads.x);
        threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
        blocks.x = (dimx[2] + threads.x - 1) / threads.x;
        blocks.y = (dimx[1] + threads.y - 1) / threads.y;
        blocks.z = layers;
        yy_ += dimy[2] * dimy[1] * position[0]
             + dimy[2] * position[1]
             + position[2];
        bmgs_cut_cuda_kernel5<<<blocks, threads>>>(
                x_, yy_, sizex, sizey, pos);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERNEL5");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}
