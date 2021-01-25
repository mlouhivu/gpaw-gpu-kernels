#include "kernels.h"

__global__ void Zcuda(bmgs_cut_cuda_kernel2)(
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

/*** New GPU implementation (multi-block) ***/
float run_kernel2(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey,
                  const unsigned int layers,
                  char *title, char *header, const int repeat)
{
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blocks, threads;

    char name[32];

    cudaEventRecord(start);
    threads.x = min(nextPow2(sizey.z), BLOCK_TOTALMAX);
    threads.y = min(nextPow2(sizey.y), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (sizey.z + threads.x - 1) / threads.x;
    blocks.y = (sizey.y + threads.y - 1) / threads.y;
    blocks.z = (sizey.x + threads.z - 1) / threads.z;
    bmgs_cut_cuda_kernel2<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERNEL2");
    if (!repeat)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}

float run_kernel2b(double *x_, const int3 sizex, const int3 pos,
                   double *y_, const int3 sizey,
                   const unsigned int layers,
                   char *title, char *header, const int repeat)
{
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blocks, threads;

    char name[32];

    cudaEventRecord(start);
    threads.x = min(nextPow2(sizey.z), BLOCK_MAX);
    threads.y = min(nextPow2(sizey.y), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (sizey.z + threads.x - 1) / threads.x;
    blocks.y = (sizey.y + threads.y - 1) / threads.y;
    blocks.z = (sizey.x + threads.z - 1) / threads.z;
    bmgs_cut_cuda_kernel2<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERN2v2");
    if (!repeat)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}
