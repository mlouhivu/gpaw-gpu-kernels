#include "kernels.h"


__global__ void Zcuda(bmgs_cut_cuda_kernel)(
        const Tcuda* a, const int3 c_sizea, Tcuda* b, const int3 c_sizeb,
        int blocks, int xdiv)
{
    int xx = gridDim.x / xdiv;  // xdiv == x ; xx == Nz
    int yy = gridDim.y / blocks; // blocks == blocks ; yy == Ny

    int blocksi = blockIdx.y / yy;  // blockID for blocks
    int i1 = (blockIdx.y - blocksi * yy) * blockDim.y + threadIdx.y;
      // gid in y from block start

    int xind = blockIdx.x / xx;  // blockID for x
    int i2 = (blockIdx.x - xind * xx) * blockDim.x + threadIdx.x;
      // gid in z from x start

    b += i2 + (i1 + (xind + blocksi * c_sizeb.x) * c_sizeb.y) * c_sizeb.z;
    a += i2 + (i1 + (xind + blocksi * c_sizea.x) * c_sizea.y) * c_sizea.z;

/*    b += i2
       + i1      * c_sizeb.z
       + xind    * c_sizeb.y * c_sizeb.z;
       + blocksi * c_sizeb.x * c_sizeb.y * c_sizeb.z;
*/
    while (xind < c_sizeb.x) {
        if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
            b[0] = a[0];
        }
        b += xdiv * c_sizeb.y * c_sizeb.z;
        a += xdiv * c_sizea.y * c_sizea.z;
        xind += xdiv;
    }
}


void Zcuda(bmgs_cut_cuda_gpu)(
        const Tcuda* a, const int sizea[3], const int starta[3],
        Tcuda* b, const int sizeb[3],
        int blocks, dim3 *blx, dim3 *thx)
{
    if (!(sizea[0] && sizea[1] && sizea[2]))
        return;

    int3 hc_sizea, hc_sizeb;
    hc_sizea.x=sizea[0];
    hc_sizea.y=sizea[1];
    hc_sizea.z=sizea[2];
    hc_sizeb.x=sizeb[0];
    hc_sizeb.y=sizeb[1];
    hc_sizeb.z=sizeb[2];

    int blockx = MIN(nextPow2(hc_sizeb.z), BLOCK_MAX);
    int blocky = MIN(
            MIN(nextPow2(hc_sizeb.y), BLOCK_TOTALMAX / blockx),
            BLOCK_MAX);
    dim3 dimBlock(blockx, blocky);
    int gridx = ((hc_sizeb.z + dimBlock.x - 1) / dimBlock.x);
    int xdiv = MAX(1, MIN(hc_sizeb.x, GRID_MAX / gridx));
    int gridy = blocks * ((hc_sizeb.y + dimBlock.y - 1) / dimBlock.y);

    gridx = xdiv * gridx;
    dim3 dimGrid(gridx, gridy);

    thx->x = blockx;
    thx->y = blocky;
    blx->x = gridx;
    blx->y = gridy;

    a += starta[2] + (starta[1] + starta[0] * hc_sizea.y) * hc_sizea.z;

    Zcuda(bmgs_cut_cuda_kernel)<<<dimGrid, dimBlock, 0>>>(
            (Tcuda*) a, hc_sizea, (Tcuda*) b, hc_sizeb,
            blocks, xdiv);
}


/*** Original GPU implementation ***/
float run_kernel0(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey,
                  const unsigned int layers,
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

    double *xx_;
    double *yy_;

    char name[32];

    cudaEventRecord(start);
    for (int i=0; i < repeat; i++) {
        xx_ = x_;
        yy_ = y_;
        bmgs_cut_cuda_gpu(xx_, dimx, position, yy_, dimy, layers,
                          &blocks, &threads);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "KERNEL0");
    if (!trial)
        sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    return time;
}
