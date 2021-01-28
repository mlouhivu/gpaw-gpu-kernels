#include "kernels.h"

__global__ void Zcuda(bmgs_cut_cuda_kernel4)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o, int blocks)
{
    int gridsize_y = (gridDim.y + blocks - 1) / blocks;
    int b = blockIdx.y / gridsize_y;
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + (blockIdx.y - b * gridsize_y) * blockDim.y;
    int tidz = threadIdx.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridsize_y * blockDim.y;
    int stridez = blockDim.z;
    int t, s, tz, sz, tb, sb;
    int i, j, k;

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

/*** New GPU implementation (multi-block, block in dim) ***/
float run_kernel4(double *x_, const int3 sizex, const int3 pos,
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
        blocks.y = layers * ((sizex.y + threads.y - 1) / threads.y);
        blocks.z = 1;
        bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
                x_, y_, sizex, sizey, pos, layers);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERNEL4");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}

float run_kernel4b(double *x_, const int3 sizex, const int3 pos,
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
        threads.x = 1;
        threads.y = MIN(nextPow2(sizex.y), BLOCK_MAX);
        threads.z = BLOCK_MAX / (threads.x * threads.y);
        blocks.x = (sizex.z + threads.x - 1) / threads.x;
        blocks.y = layers * ((sizex.y + threads.y - 1) / threads.y);
        blocks.z = 1;
        bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
                x_, y_, sizex, sizey, pos, layers);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERN4v2");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}
