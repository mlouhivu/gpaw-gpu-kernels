#include "kernels.h"

__global__ void Zcuda(bmgs_cut_cuda_kernel1)(
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

/*** New GPU implementation ***/
float run_kernel1(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey, const int layers,
                  char *title, char *header,
                  const int repeat, const int trial)
{
    const int n = sizex.x * sizex.y * sizex.z;
    const int m = sizey.x * sizey.y * sizey.z;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blocks, threads;

    double *xx_;
    double *yy_;

    char name[32];

    cudaEventRecord(start);
    for (int i=0; i < repeat; i++) {
        xx_ = x_;
        yy_ = y_;
        threads.x = min(nextPow2(sizey.z), BLOCK_TOTALMAX);
        threads.y = min(nextPow2(sizey.y), BLOCK_TOTALMAX / threads.x);
        threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
        blocks.x = (sizey.z + threads.x - 1) / threads.x;
        blocks.y = (sizey.y + threads.y - 1) / threads.y;
        blocks.z = (sizey.x + threads.z - 1) / threads.z;
        for (int l=0; l < layers; l++) {
            bmgs_cut_cuda_kernel1<<<blocks, threads>>>(
                    xx_, yy_, sizex, sizey, pos);
            xx_ += n;
            yy_ += m;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERNEL1");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}
