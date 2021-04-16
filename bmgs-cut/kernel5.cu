#include "kernels.h"

__global__ void Zcuda(bmgs_cut_cuda_kernel5)(
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
        sz = sb + n.z * n.y * i;
        for (j = tidy; j < m.y; j += stridey) {
            t = tz + m.z * j;
            s = sz + n.z * j;
            for (k = tidx; k < m.z; k += stridex) {
                tgt[k + t] = MULTT(phase, src[k + s]);
            }
        }
    }
}

/*** New GPU implementation (multi-block, block in dim) ***/
float run_kernel5(Tcuda *x_, const int3 sizex, const int3 pos,
                  Tcuda *y_, const int3 sizey, const int layers,
                  const Tcuda phase_,
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

    Tcuda *xx_;

    char name[32];

    cudaEventRecord(start);
    for (int i=0; i < repeat; i++) {
        xx_ = x_;
        threads.x = MIN(nextPow2(dimy[2]), BLOCK_TOTALMAX);
        threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
        threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
        blocks.x = (dimy[2] + threads.x - 1) / threads.x;
        blocks.y = (dimy[1] + threads.y - 1) / threads.y;
        blocks.z = layers;
        xx_ += dimx[2] * dimx[1] * position[0]
             + dimx[2] * position[1]
             + position[2];
        Zcuda(bmgs_cut_cuda_kernel5)<<<blocks, threads>>>(
                xx_, y_, sizex, sizey, pos, phase_);
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
