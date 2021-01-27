#include "kernels.h"

__global__ void Zcuda(bmgs_paste_cuda_kernel)(
        const double* a, const int3 c_sizea, double* b, const int3 c_sizeb,
        int blocks, int xdiv)
{
    int xx = gridDim.x / xdiv;
    int yy = gridDim.y / blocks;

    int blocksi = blockIdx.y / yy;
    int i1 = (blockIdx.y - blocksi * yy) * blockDim.y + threadIdx.y;

    int xind = blockIdx.x / xx;
    int i2 = (blockIdx.x - xind * xx) * blockDim.x + threadIdx.x;

    b += i2 + (i1 + (xind + blocksi * c_sizeb.x) * c_sizeb.y) * c_sizeb.z;
    a += i2 + (i1 + (xind + blocksi * c_sizea.x) * c_sizea.y) * c_sizea.z;

    while (xind < c_sizea.x) {
        if ((i2 < c_sizea.z) && (i1 < c_sizea.y)) {
            b[0] = a[0];
        }
        b += xdiv * c_sizeb.y * c_sizeb.z;
        a += xdiv * c_sizea.y * c_sizea.z;
        xind += xdiv;
    }
}

__global__ void Zcuda(bmgs_paste_zero_cuda_kernel)(
        const Tcuda* a, const int3 c_sizea, Tcuda* b, const int3 c_sizeb,
        const int3 c_startb, const int3 c_blocks_bc, int blocks)
{
    int xx = gridDim.x / XDIV;
    int yy = gridDim.y / blocks;

    int blocksi = blockIdx.y / yy;
    int i1bl = blockIdx.y - blocksi * yy;
    int i1tid = threadIdx.y;
    int i1 = i1bl * BLOCK_SIZEY + i1tid;

    int xind = blockIdx.x / xx;
    int i2bl = blockIdx.x - xind * xx;
    int i2tid = threadIdx.x;
    int i2 = i2bl * BLOCK_SIZEX + i2tid;

    int xlen = (c_sizea.x + XDIV - 1) / XDIV;
    int xstart = xind * xlen;
    int xend = MIN(xstart + xlen, c_sizea.x);

    b += c_sizeb.x * c_sizeb.y * c_sizeb.z * blocksi;
    a += c_sizea.x * c_sizea.y * c_sizea.z * blocksi;

    if (xind==0) {
        Tcuda *bb = b + i2 + i1 * c_sizeb.z;
#pragma unroll 3
        for (int i0=0; i0 < c_startb.x; i0++) {
            if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
                bb[0] = MAKED(0);
            }
            bb += c_sizeb.y * c_sizeb.z;
        }
    }
    if (xind == XDIV - 1) {
        Tcuda *bb = b + (c_startb.x + c_sizea.x) * c_sizeb.y * c_sizeb.z
                  + i2 + i1 * c_sizeb.z;
#pragma unroll 3
        for (int i0 = c_startb.x + c_sizea.x; i0 < c_sizeb.x; i0++) {
            if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
                bb[0] = MAKED(0);
            }
            bb += c_sizeb.y * c_sizeb.z;
        }
    }

    int i1blbc = gridDim.y / blocks - i1bl - 1;
    int i2blbc = gridDim.x / XDIV - i2bl - 1;

    if (i1blbc<c_blocks_bc.y || i2blbc<c_blocks_bc.z) {
        int i1bc = i1blbc * BLOCK_SIZEY + i1tid;
        int i2bc = i2blbc * BLOCK_SIZEX + i2tid;

        b += (c_startb.x + xstart) * c_sizeb.y * c_sizeb.z;
        for (int i0=xstart; i0 < xend; i0++) {
            if ((i1bc < c_startb.y) && (i2 < c_sizeb.z)) {
                b[i2 + i1bc * c_sizeb.z] = MAKED(0);
            }
            if ((i1bc + c_sizea.y + c_startb.y < c_sizeb.y)
                    && (i2 < c_sizeb.z)) {
                b[i2 + i1bc * c_sizeb.z
                  + (c_sizea.y + c_startb.y) * c_sizeb.z] = MAKED(0);
            }
            if ((i2bc < c_startb.z) && (i1 < c_sizeb.y)) {
                b[i2bc + i1 * c_sizeb.z] = MAKED(0);
            }
            if ((i2bc + c_sizea.z + c_startb.z < c_sizeb.z)
                    && (i1 < c_sizeb.y)) {
                b[i2bc + i1 * c_sizeb.z + c_sizea.z + c_startb.z] = MAKED(0);
            }
            b += c_sizeb.y * c_sizeb.z;
        }
    } else {
        b += c_startb.z + (c_startb.y + c_startb.x * c_sizeb.y) * c_sizeb.z;

        b += i2 + i1 * c_sizeb.z + xstart * c_sizeb.y * c_sizeb.z;
        a += i2 + i1 * c_sizea.z + xstart * c_sizea.y * c_sizea.z;
        for (int i0=xstart; i0 < xend; i0++) {
            if ((i2 < c_sizea.z) && (i1 < c_sizea.y)) {
                b[0] = a[0];
            }
            b += c_sizeb.y * c_sizeb.z;
            a += c_sizea.y * c_sizea.z;
        }
    }
}

void Zcuda(bmgs_paste_cuda_gpu)(
        const Tcuda* a, const int sizea[3],
        Tcuda* b, const int sizeb[3], const int startb[3],
        int blocks, dim3 *blx, dim3 *thx)
{
    if (!(sizea[0] && sizea[1] && sizea[2]))
        return;

    int3 hc_sizea, hc_sizeb;
    hc_sizea.x = sizea[0];
    hc_sizea.y = sizea[1];
    hc_sizea.z = sizea[2] * sizeof(Tcuda) / sizeof(double);
    hc_sizeb.x = sizeb[0];
    hc_sizeb.y = sizeb[1];
    hc_sizeb.z = sizeb[2] * sizeof(Tcuda) / sizeof(double);

    int blockx = MIN(nextPow2(hc_sizea.z), BLOCK_MAX);
    int blocky = MIN(
            MIN(nextPow2(hc_sizea.y), BLOCK_TOTALMAX / blockx),
            BLOCK_MAX);
    dim3 dimBlock(blockx, blocky);
    int gridx = ((hc_sizea.z + dimBlock.x - 1) / dimBlock.x);
    int xdiv = MAX(1, MIN(hc_sizea.x, GRID_MAX / gridx));
    int gridy = blocks * ((hc_sizea.y + dimBlock.y - 1) / dimBlock.y);

    gridx = xdiv * gridx;
    dim3 dimGrid(gridx, gridy);

    thx->x = dimBlock.x;
    thx->y = dimBlock.y;
    blx->x = dimGrid.x;
    blx->y = dimGrid.y;

    b += startb[2] + (startb[1] + startb[0] * sizeb[1]) * sizeb[2];

    Zcuda(bmgs_paste_cuda_kernel)<<<dimGrid, dimBlock>>>(
            (double*) a, hc_sizea, (double*) b, hc_sizeb, blocks, xdiv);
}

void Zcuda(bmgs_paste_zero_cuda_gpu)(
        const Tcuda* a, const int sizea[3],
        Tcuda* b, const int sizeb[3], const int startb[3],
        int blocks, dim3 *blx, dim3 *thx)
{
    if (!(sizea[0] && sizea[1] && sizea[2]))
        return;

    int3 bc_blocks;
    int3 hc_sizea, hc_sizeb, hc_startb;
    hc_sizea.x = sizea[0];
    hc_sizea.y = sizea[1];
    hc_sizea.z = sizea[2];
    hc_sizeb.x = sizeb[0];
    hc_sizeb.y = sizeb[1];
    hc_sizeb.z = sizeb[2];
    hc_startb.x = startb[0];
    hc_startb.y = startb[1];
    hc_startb.z = startb[2];

    bc_blocks.y = hc_sizeb.y - hc_sizea.y > 0
                ? MAX((hc_sizeb.y - hc_sizea.y + BLOCK_SIZEY - 1)
                        / BLOCK_SIZEY, 1)
                : 0;
    bc_blocks.z = hc_sizeb.z - hc_sizea.z > 0
                ? MAX((hc_sizeb.z - hc_sizea.z + BLOCK_SIZEX - 1)
                        / BLOCK_SIZEX, 1)
                : 0;

    int gridy = blocks * ((sizeb[1] + BLOCK_SIZEY - 1) / BLOCK_SIZEY
                          + bc_blocks.y);
    int gridx = XDIV * ((sizeb[2] + BLOCK_SIZEX - 1) / BLOCK_SIZEX
                        + bc_blocks.z);

    dim3 dimBlock(BLOCK_SIZEX, BLOCK_SIZEY);
    dim3 dimGrid(gridx, gridy);

    thx->x = dimBlock.x;
    thx->y = dimBlock.y;
    blx->x = dimGrid.x;
    blx->y = dimGrid.y;

    Zcuda(bmgs_paste_zero_cuda_kernel)<<<dimGrid, dimBlock>>>(
            (Tcuda*) a, hc_sizea, (Tcuda*) b, hc_sizeb, hc_startb,
            bc_blocks, blocks);
}


/*** Original GPU implementation ***/
float run_kernel0(double *x_, const int3 sizex, const int3 pos,
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

    double *xx_;
    double *yy_;

    char name[32];

    cudaEventRecord(start);
    for (int i=0; i < repeat; i++) {
        xx_ = x_;
        yy_ = y_;
        bmgs_paste_cuda_gpu(xx_, dimx, yy_, dimy, position, layers,
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
