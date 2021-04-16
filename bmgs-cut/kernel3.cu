#include "kernels.h"

__global__ void Zcuda(bmgs_cut_cuda_kernel3)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o, const Tcuda phase)
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
                tgt[k + t] = MULTT(phase, src[k + s]);
            }
        }
    }
}

/*** New GPU implementation (multi-block, block in dim) ***/
float run_kernel3(Tcuda *x_, const int3 sizex, const int3 pos,
                  Tcuda *y_, const int3 sizey, const int layers,
                  const Tcuda phase_,
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
        threads.x = MIN(nextPow2(sizey.z), BLOCK_TOTALMAX);
        threads.y = MIN(nextPow2(sizey.y), BLOCK_TOTALMAX / threads.x);
        threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
        blocks.x = (sizey.z + threads.x - 1) / threads.x;
        blocks.y = (sizey.y + threads.y - 1) / threads.y;
        blocks.z = layers;
        Zcuda(bmgs_cut_cuda_kernel3)<<<blocks, threads>>>(
                x_, y_, sizex, sizey, pos, phase_);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERNEL3");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}

float run_kernel3b(Tcuda *x_, const int3 sizex, const int3 pos,
                   Tcuda *y_, const int3 sizey, const int layers,
                   const Tcuda phase_,
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
        threads.x = MIN(nextPow2(sizey.z), BLOCK_MAX);
        threads.y = MIN(nextPow2(sizey.y), BLOCK_TOTALMAX / threads.x);
        threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
        blocks.x = (sizey.z + threads.x - 1) / threads.x;
        blocks.y = (sizey.y + threads.y - 1) / threads.y;
        blocks.z = layers;
        Zcuda(bmgs_cut_cuda_kernel3)<<<blocks, threads>>>(
                x_, y_, sizex, sizey, pos, phase_);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERN3v2");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}

float run_kernel3c(Tcuda *x_, const int3 sizex, const int3 pos,
                   Tcuda *y_, const int3 sizey, const int layers,
                   const Tcuda phase_,
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
        threads.x = MIN(nextPow2(sizey.z), BLOCK_MAX);
        threads.y = MIN(nextPow2(sizey.y), BLOCK_MAX / threads.x);
        threads.z = BLOCK_MAX / (threads.x * threads.y);
        blocks.x = (sizey.z + threads.x - 1) / threads.x;
        blocks.y = (sizey.y + threads.y - 1) / threads.y;
        blocks.z = layers;
        Zcuda(bmgs_cut_cuda_kernel3)<<<blocks, threads>>>(
                x_, y_, sizex, sizey, pos, phase_);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERN3v3");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}
