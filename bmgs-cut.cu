#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_MAX 32
#define GRID_MAX 65535
#define BLOCK_TOTALMAX 256

#define Tcuda          double
#define Zcuda(f)       f
#define MULTT(a,b)     ((a) * (b))
#define MULTD(a,b)     ((a) * (b))
#define MULDT(a,b)     ((a) * (b))
#define ADD(a,b)       ((a) + (b))
#define ADD3(a,b,c)    ((a) + (b) + (c))
#define ADD4(a,b,c,d)  ((a) + (b) + (c) + (d))
#define IADD(a,b)      ((a) += (b))
#define MAKED(a)       (a)
#define CONJ(a)        (a)
#define REAL(a)        (a)
#define IMAG(a)        (0)
#define NEG(a)         (-(a))

#define MAX(a,b)  (((a) > (b)) ? (a) : (b))
#define MIN(a,b)  (((a) < (b)) ? (a) : (b))


static unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void bmgs_cut(const double *a, const int n[3], const int c[3],
              double *b, const int m[3])
{
  a += c[2] + (c[1] + c[0] * n[1]) * n[2];
  for (int i0 = 0; i0 < m[0]; i0++)
    {
      for (int i1 = 0; i1 < m[1]; i1++)
        {
          memcpy(b, a, m[2] * sizeof(double));
          a += n[2];
          b += m[2];
        }
      a += n[2] * (n[1] - m[1]);
    }
}


__global__ void Zcuda(bmgs_cut_cuda_kernel6)(
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
    for (i = tidz; i < m.x; i += stridez) {
        tz = tb + m.z * m.y * i;
        sz = sb + n.z * n.y * i;
        for (j = tidy; j < m.y; j += stridey) {
            t = tz + m.z * j;
            s = sz + n.z * j;
            for (k = tidx; k < m.z; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}

__global__ void Zcuda(bmgs_cut_cuda_kernel5)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o, int blocks)
{
    int gridsize_y = gridDim.y / blocks;
    int b = blockIdx.y / gridsize_y;
    int tidy = threadIdx.y + (blockIdx.y - b * gridsize_y) * blockDim.y;
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidz = threadIdx.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
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

__global__ void Zcuda(bmgs_cut_cuda_kernel4)(
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

__global__ void Zcuda(bmgs_cut_cuda_kernel4b)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = blockDim.z;
    int b = blockIdx.z;
    int t, s, tb, sb;
    int i, j, k;

    tb = m.z * m.y * m.x * b;
    sb = n.z * n.y * n.x * b
       + n.z * n.y * o.x
       + n.z * o.y
       + o.z;
    for (i = tidz; i < m.x; i += stridez) {
        for (j = tidy; j < m.y; j += stridey) {
            t = tb + m.z * m.y * i + m.z * j;
            s = sb + n.z * n.y * i + n.z * j;
            for (k = tidx; k < m.z; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}

__global__ void Zcuda(bmgs_cut_cuda_kernel3)(
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

__global__ void Zcuda(bmgs_cut_cuda_kernel2)(
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

__global__ void Zcuda(bmgs_cut_cuda_kernel2b)(
        Tcuda *src, Tcuda *tgt, int3 n, int3 m, int3 o)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int t, s, tz, sz;
    int i;

    for (; tidy < m.x; tidy += stridey) {
        t = m.z * m.y * tidy;
        s = n.z * n.y * (tidy + o.x) + o.z;
        for (; tidx < m.y; tidx += stridex) {
            t = tz + m.z * tidx;
            s = sz + n.z * (tidy + o.y);
            for (i=0; i < m.z; i++) {
                tgt[tidx + t + i] = src[tidx + s + i];
            }
        }
    }
}



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

#ifdef DEBUG_CUDA_CUT
    int ng = sizea[0] * sizea[1] * sizea[2];
    int ng2 = sizeb[0] * sizeb[1] * sizeb[2];
    double* a_cpu = GPAW_MALLOC(double, ng * blocks);
    double* b_cpu = GPAW_MALLOC(double, ng2 * blocks);
    double* a_cpu2 = GPAW_MALLOC(double, ng * blocks);
    double* b_cpu2 = GPAW_MALLOC(double, ng2 * blocks);
    const Tcuda* a2 = a;

    GPAW_CUDAMEMCPY(a_cpu, a, double, ng * blocks, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(b_cpu, b, double, ng2 * blocks, cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_CUT

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
    //gpaw_cudaSafeCall(cudaGetLastError());

#ifdef DEBUG_CUDA_CUT
    for (int m=0; m < blocks; m++) {
        bmgs_cut(a_cpu + m * ng, sizea, starta, b_cpu + m * ng2, sizeb);
    }
    cudaDeviceSynchronize();
    GPAW_CUDAMEMCPY(a_cpu2, a2, double, ng * blocks,
            cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(b_cpu2, b, double, ng2 * blocks,
            cudaMemcpyDeviceToHost);

    double a_err = 0;
    double b_err = 0;
    for (int i=0; i < ng2 * blocks; i++) {
        b_err = MAX(b_err, fabs(b_cpu[i] - b_cpu2[i]));
        if (i < ng * blocks) {
            a_err = MAX(a_err, fabs(a_cpu[i] - a_cpu2[i]));
        }
    }
    if ((b_err > GPAW_CUDA_ABS_TOL_EXCT)
            || (a_err > GPAW_CUDA_ABS_TOL_EXCT)) {
        fprintf(stderr, "Debug cuda cut errors: a %g b %g\n",
                a_err, b_err);
    }
    free(a_cpu);
    free(b_cpu);
    free(a_cpu2);
    free(b_cpu2);
#endif //DEBUG_CUDA_CUT
}

int main(void)
{
    int i, j, k, l, s, t;
    // carbon nanotube
    /*
    const int layers = 56;
    const int dimx[3] = {41,21,32};
    const int dimy[3] = {41,21,1};
    int position[3] = {0,0,0};

    const int layers = 56;
    const int dimx[3] = {85,46,68};
    const int dimy[3] = {79,40,3};
    int position[3] = {3,3,62};

    const int layers = 56;
    const int dimx[3] = {85,45,68};
    const int dimy[3] = {79,39,3};
    int position[3] = {3,3,3};

    const int layers = 56;
    const int dimx[3] = {21,11,17};
    const int dimy[3] = {19,1,15};
    int position[3] = {1,9,1};

    const int layers = 56;
    const int dimx[3] = {21,11,18};
    const int dimy[3] = {19,9,1};
    int position[3] = {1,1,1};

    // copper filament
    const int layers = 25;
    const int dimx[3] = {89,52,62};
    const int dimy[3] = {83,46,3};
    int position[3] = {3,3,56};

    const int layers = 25;
    const int dimx[3] = {43,24,29};
    const int dimy[3] = {43,24,1};
    int position[3] = {0,0,0};

    const int layers = 25;
    const int dimx[3] = {43,25,30};
    const int dimy[3] = {41,24,28};
    int position[3] = {1,1,1};

    const int layers = 48;
    const int dimx[3] = {89,52,62};
    const int dimy[3] = {83,46,3};
    int position[3] = {3,3,56};

    // single fullerene
    const int layers = 1;
    const int dimx[3] = {6,7,12};
    const int dimy[3] = {1,5,11};
    int position[3] = {0,1,0};

    const int layers = 1;
    const int dimx[3] = {12,12,23};
    const int dimy[3] = {1,11,22};
    int position[3] = {0,0,0};

    // other
    const int layers = 12;
    const int dimx[3] = {252,31,64};
    const int dimy[3] = {252,31,1};
    int position[3] = {0,0,0};
    */

    const int layers = 8;
    const int dimx[3] = {100,100,100};
    const int dimy[3] = {10,10,10};
    int position[3] = {22,44,66};
    /*
    */

    const int n = dimx[0] * dimx[1] * dimx[2];
    const int m = dimy[0] * dimy[1] * dimy[2];
    double x[layers * n], y[layers * m], y_ref[layers * m];
    double *xp = &x[0];
    double *yp = &y[0];
    double *x_, *y_;
    double *xx_, *yy_;

    dim3 blocks(32, 32, 32);
    dim3 threads(16, 16, 1);
    //dim3 blocks(32, 32, 32);
    //dim3 threads(32, 8, 1);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float results[8];
    int ri=0;

    int3 sizex, sizey, pos;
    sizex.x = dimx[0];
    sizex.y = dimx[1];
    sizex.z = dimx[2];
    sizey.x = dimy[0];
    sizey.y = dimy[1];
    sizey.z = dimy[2];

    // initialise data
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    // copy reference values
    for (l=0; l < layers; l++) {
        for (i=0; i < dimy[0]; i++) {
            for (j=0; j < dimy[1]; j++) {
                for (k=0; k < dimy[2]; k++) {
                    t = dimy[2] * dimy[1] * dimy[0] * l
                      + dimy[2] * dimy[1] * i
                      + dimy[2] * j
                      + k;
                    s = dimx[2] * dimx[1] * dimx[0] * l
                      + dimx[2] * dimx[1] * (i + position[0])
                      + dimx[2] * (j + position[1])
                      + k + position[2];
                    y_ref[t] = x[s];
                }
            }
        }
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n * layers);
    cudaMalloc((void **) &y_, sizeof(double) * m * layers);
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** CPU implementation ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    printf("\nCPU\n");
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaEventRecord(start);
    for (l=0; l < layers; l++) {
        bmgs_cut(xp + l * n, dimx, position, yp + l * m, dimy);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** Original GPU implementation ***/
    xx_ = x_;
    yy_ = y_;
    dim3 blx, thx;
    cudaEventRecord(start);
    bmgs_cut_cuda_gpu(xx_, dimx, position, yy_, dimy, layers, &blx, &thx);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blx.x, blx.y, blx.z, thx.x, thx.y, thx.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

#ifdef MY_VERBOSE
    /*** New GPU implementation ***/
    xx_ = x_;
    yy_ = y_;
    cudaEventRecord(start);
    for (l=0; l < layers; l++) {
        pos.x = position[0]; // + dimx[0] * l;
        pos.y = position[1];
        pos.z = position[2];
        bmgs_cut_cuda_kernel2<<<blocks, threads>>>(xx_, yy_, sizex, sizey, pos);
        /*dim3 blx(32, 32, 1);
        dim3 thx(32, 8, 1);
        bmgs_cut_cuda_kernel2b<<<blx, thx>>>(xx_, yy_, sizex, sizey, pos);
        */xx_ += n;
        yy_ += m;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\nKERNEL2\n");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (multi-block) ***/
    cudaEventRecord(start);
    bmgs_cut_cuda_kernel3<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL3\n");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (multi-block, block in dim) ***/
    dim3 blocks4(32, 32, layers);
    cudaEventRecord(start);
    bmgs_cut_cuda_kernel4<<<blocks4, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL4\n");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks4.x, blocks4.y, blocks4.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (multi-block, block in dim) ***/
    dim3 blocks5(32, 32 * layers, 1);
    cudaEventRecord(start);
    bmgs_cut_cuda_kernel5<<<blocks5, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL5\n");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks5.x, blocks5.y, blocks5.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (multi-block, block in dim) ***/
    dim3 blocks6(32, 32, layers);
    xx_ = x_;
    cudaEventRecord(start);
    xx_ += dimx[2] * dimx[1] * position[0]
         + dimx[2] * position[1]
         + position[2];
    bmgs_cut_cuda_kernel6<<<blocks6, threads>>>(
            xx_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL6\n");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks6.x, blocks6.y, blocks6.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;
#endif

    /*** OPTIMISED KERNELS ***/

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised) ***/
    xx_ = x_;
    yy_ = y_;
    cudaEventRecord(start);
    threads.x = min(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = min(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = (dimy[0] + threads.z - 1) / threads.z;
    for (l=0; l < layers; l++) {
        pos.x = position[0];
        pos.y = position[1];
        pos.z = position[2];
        bmgs_cut_cuda_kernel2<<<blocks, threads>>>(xx_, yy_, sizex, sizey, pos);
        xx_ += n;
        yy_ += m;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\nKERNEL2 (optimised)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised multi-block) ***/
    cudaEventRecord(start);
    threads.x = min(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = min(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = (dimy[0] + threads.z - 1) / threads.z;
    bmgs_cut_cuda_kernel3<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL3 (optimised)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised multi-block) ***/
    cudaEventRecord(start);
    threads.x = min(nextPow2(dimy[2]), BLOCK_MAX);
    threads.y = min(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = (dimy[0] + threads.z - 1) / threads.z;
    bmgs_cut_cuda_kernel3<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL3 (optimised v2)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = layers;
    bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL4 (optimised)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_MAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = layers;
    bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL4 (optimised v2)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_MAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_MAX / threads.x);
    threads.z = BLOCK_MAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = layers;
    bmgs_cut_cuda_kernel4<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL4 (optimised v3)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

#ifdef MY_VERBOSE
    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = layers * (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = 1;
    bmgs_cut_cuda_kernel5<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL5 (optimised)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    cudaEventRecord(start);
    threads.x = 1;
    threads.y = MIN(MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x),
                    BLOCK_MAX);
    threads.z = BLOCK_MAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = layers * (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = 1;
    bmgs_cut_cuda_kernel5<<<blocks, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL5 (optimised v2)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;

    /*** reset ***/
    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);

    /*** New GPU implementation (optimised multi-block, block in dim) ***/
    xx_ = x_;
    cudaEventRecord(start);
    threads.x = MIN(nextPow2(dimy[2]), BLOCK_TOTALMAX);
    threads.y = MIN(nextPow2(dimy[1]), BLOCK_TOTALMAX / threads.x);
    threads.z = BLOCK_TOTALMAX / (threads.x * threads.y);
    blocks.x = (dimy[2] + threads.x - 1) / threads.x;
    blocks.y = (dimy[1] + threads.y - 1) / threads.y;
    blocks.z = layers;
    xx_ += dimx[2] * dimx[1] * position[0]
         + dimx[2] * position[1]
         + position[2];
    bmgs_cut_cuda_kernel6<<<blocks, threads>>>(
            xx_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nKERNEL6 (optimised)");
    printf("  <<<(%d,%d,%d), (%d, %d, %d)>>>\n",
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    printf("exec time: %f\n", time);
    results[ri++] = time;
#endif

    printf("\nTiming results:\n");
    for (i=0; i < ri; i++) {
        printf("%f ", results[i]);
    }
    printf("\n\n");

    return 0;
}
