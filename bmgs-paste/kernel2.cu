#include "kernels.h"

__global__ void Zcuda(bmgs_paste_cuda_kernel2)(
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
        for (i = tidz; i < n.x; i += stridez) {
            tz = tb + m.z * m.y * (i + o.x) + o.z;
            sz = sb + n.z * n.y * i;
            for (j = tidy; j < n.y; j += stridey) {
                t = tz + m.z * (j + o.y);
                s = sz + n.z * j;
                for (k = tidx; k < n.z; k += stridex) {
                    tgt[k + t] = src[k + s];
                }
            }
        }
    }
}

/*** New GPU implementation (multi-block) ***/
float run_kernel2(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey, const int layers,
                  char *title, char *header,
                  const int repeat, const int trial)
{
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blocks, threads;

    char name[32];

    cudaEventRecord(start);
    for (int i=0; i < repeat; i++) {
        threads.x = MIN(nextPow2(sizex.z), BLOCK_TOTALMAX);
        threads.y = MIN(nextPow2(sizex.y), BLOCK_TOTALMAX / threads.x);
        threads.z = MIN(BLOCK_TOTALMAX / (threads.x * threads.y), BLOCK_MAX);
        blocks.x = (sizex.z + threads.x - 1) / threads.x;
        blocks.y = (sizex.y + threads.y - 1) / threads.y;
        blocks.z = (sizex.x + threads.z - 1) / threads.z;
        bmgs_paste_cuda_kernel2<<<blocks, threads>>>(
                x_, y_, sizex, sizey, pos, layers);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERNEL2");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}

float run_kernel2b(double *x_, const int3 sizex, const int3 pos,
                   double *y_, const int3 sizey, const int layers,
                   char *title, char *header,
                   const int repeat, const int trial)
{
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blocks, threads;

    char name[32];

    cudaEventRecord(start);
    for (int i=0; i < repeat; i++) {
        threads.x = MIN(nextPow2(sizex.z), BLOCK_MAX);
        threads.y = MIN(MIN(nextPow2(sizex.y), BLOCK_TOTALMAX / threads.x),
                        BLOCK_MAX);
        threads.z = MIN(BLOCK_TOTALMAX / (threads.x * threads.y), BLOCK_MAX);
        blocks.x = (sizex.z + threads.x - 1) / threads.x;
        blocks.y = (sizex.y + threads.y - 1) / threads.y;
        blocks.z = (sizex.x + threads.z - 1) / threads.z;
        bmgs_paste_cuda_kernel2<<<blocks, threads>>>(
                x_, y_, sizex, sizey, pos, layers);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERN2v2");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}
