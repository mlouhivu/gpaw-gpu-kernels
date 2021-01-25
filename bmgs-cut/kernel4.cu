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

/*** New GPU implementation (multi-block, block in dim) ***/
float run_kernel4(double *x_, const int3 sizex, const int3 pos,
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
    threads.x = MIN(nextPow2(sizey.z), BLOCK_MAX);
    threads.y = MIN(MIN(nextPow2(sizey.y), BLOCK_TOTALMAX / threads.x),
                    BLOCK_MAX);
    threads.z = MIN(BLOCK_TOTALMAX / (threads.x * threads.y), BLOCK_MAX);
    blocks.x = (sizey.z + threads.x - 1) / threads.x;
    blocks.y = layers * ((sizey.y + threads.y - 1) / threads.y);
    blocks.z = 1;
    bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERNEL4");
    if (!repeat)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}

float run_kernel4b(double *x_, const int3 sizex, const int3 pos,
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
    threads.x = 1;
    threads.y = MIN(nextPow2(sizey.y), BLOCK_MAX);
    threads.z = MIN(BLOCK_MAX / (threads.x * threads.y), BLOCK_MAX);
    blocks.x = (sizey.z + threads.x - 1) / threads.x;
    blocks.y = layers * ((sizey.y + threads.y - 1) / threads.y);
    blocks.z = 1;
    bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERN4v2");
    if (!repeat)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}
