#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_MAX 32
#define GRID_MAX 65535
#define BLOCK_TOTALMAX 256

//#define NONOPT_KERNELS 1

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

double variance(double *reference, double *result, int n)
{
    int i;
    double error = 0.0;
    double diff;

    for (i=0; i < n; i++) {
        diff = reference[i] - result[i];
        error += diff * diff;
    }
    return sqrt(error) / n;
}

void check_result(const char *name, double *y_ref, double *y, int n,
                  double time, int verbose)
{
    double error;

    error = variance(y_ref, y, n);
    if (error || verbose) {
        printf("\n%s\n", name);
        printf("reference: %f %f %f ... %f %f\n",
                y_ref[0], y_ref[1], y_ref[2], y_ref[n - 2], y_ref[n - 1]);
        printf("   result: %f %f %f ... %f %f\n",
                y[0], y[1], y[2], y[n - 2], y[n - 1]);
        printf(" variance: %f\n", error);
        printf("exec time: %f\n", time);
    }
}

int run(const unsigned int layers, const int3 sizex, const int3 sizey,
        int3 pos, float *results, char *title)
{
    int i, j, k, l, s, t;
    int verbose = 0;
    char header[512];
    char name[32];

    const int dimx[3] = {sizex.x, sizex.y, sizex.z};
    const int dimy[3] = {sizey.x, sizey.y, sizey.z};
    const int position[3] = {pos.x, pos.y, pos.z};

    const int n = dimx[0] * dimx[1] * dimx[2];
    const int m = dimy[0] * dimy[1] * dimy[2];
    double x[layers * n], y[layers * m], y_ref[layers * m];
    double *xp = &x[0];
    double *yp = &y[0];
    double *x_, *y_;
    double *xx_, *yy_;

    dim3 blocks(32, 32, 32);
    dim3 threads(16, 16, 1);
    dim3 blx, thx;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ri=0;

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
    cudaEventRecord(start);
    for (l=0; l < layers; l++) {
        bmgs_cut(xp + l * n, dimx, position, yp + l * m, dimy);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "CPU");
    sprintf(title, "%8s", name);
    check_result(name, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaEventRecord(start);
    bmgs_cut_cuda_gpu(xx_, dimx, position, yy_, dimy, layers, &blx, &thx);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blx.x, blx.y, blx.z, thx.x, thx.y, thx.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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

#ifdef NONOPT_KERNELS
    /*** New GPU implementation ***/
    xx_ = x_;
    yy_ = y_;
    cudaEventRecord(start);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL3");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    blx = {32, 32, layers};
    cudaEventRecord(start);
    bmgs_cut_cuda_kernel4<<<blx, threads>>>(
            x_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL4");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blx.x, blx.y, blx.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    blx = {32, 32 * layers, 1};
    cudaEventRecord(start);
    bmgs_cut_cuda_kernel5<<<blx, threads>>>(
            x_, y_, sizex, sizey, pos, layers);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL5");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blx.x, blx.y, blx.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    blx = {32, 32, layers};
    xx_ = x_;
    cudaEventRecord(start);
    xx_ += dimx[2] * dimx[1] * position[0]
         + dimx[2] * position[1]
         + position[2];
    bmgs_cut_cuda_kernel6<<<blx, threads>>>(
            xx_, y_, sizex, sizey, pos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL6");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blx.x, blx.y, blx.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL3");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERN3v2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL4");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERN4v2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERN4v3");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;

#ifdef DEFUNC
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL5");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL5v2");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
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
    cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
    sprintf(name, "KERNEL6");
    sprintf(title, "%s %8s", title, name);
    sprintf(header, "%s (optimised)  <<<(%d,%d,%d), (%d, %d, %d)>>>", name,
            blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    check_result(header, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time;
#endif

    return ri;
}

int main(void)
{
    int i, j;

    int ri = 0;
    int rj = 0;
    int trials = 512;
    int kernels = 512;
    float results[trials][kernels];
    float best[trials];
    float total[kernels];
    float time;

    char title[512];

    int layers;
    int3 dimx;
    int3 dimy;
    int3 position;

    for (i=0; i < kernels; i++)
        total[i] = 0.0;

    // carbon nanotube
    layers = 56;
    dimx = {41,21,32};
    dimy = {41,21,1};
    position = {0,0,0};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 56;
    dimx = {85,46,68};
    dimy = {79,40,3};
    position = {3,3,62};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 56;
    dimx = {85,45,68};
    dimy = {79,39,3};
    position = {3,3,3};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 56;
    dimx = {21,11,17};
    dimy = {19,1,15};
    position = {1,9,1};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 56;
    dimx = {21,11,18};
    dimy = {19,9,1};
    position = {1,1,1};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    // copper filament
    layers = 25;
    dimx = {89,52,62};
    dimy = {83,46,3};
    position = {3,3,56};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 25;
    dimx = {43,24,29};
    dimy = {43,24,1};
    position = {0,0,0};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 25;
    dimx = {43,25,30};
    dimy = {41,24,28};
    position = {1,1,1};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 48;
    dimx = {89,52,62};
    dimy = {83,46,3};
    position = {3,3,56};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    // single fullerene
    layers = 1;
    dimx = {6,7,12};
    dimy = {1,5,11};
    position = {0,1,0};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 1;
    dimx = {12,12,23};
    dimy = {1,11,22};
    position = {0,0,0};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    // other
    layers = 12;
    dimx = {252,31,64};
    dimy = {252,31,1};
    position = {0,0,0};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    layers = 8;
    dimx = {100,100,100};
    dimy = {10,10,10};
    position = {22,44,66};
    j = run(layers, dimx, dimy, position, results[ri++], title);
    rj = MAX(rj, j);

    printf("\nTiming results:\n%s\n", title);
    for (i=0; i < ri; i++) {
        best[i] = 9999.0;
        for (j=0; j < rj; j++) {
            printf("%f ", results[i][j]);
            best[i] = MIN(best[i], results[i][j]);
        }
        printf("\n");
    }
    printf("\nCompared to the best:\n%s\n", title);
    for (i=0; i < ri; i++) {
        for (j=0; j < rj; j++) {
            time = results[i][j] - best[i];
            if (time > 0.0)
                printf("%f ", time);
            else
                printf("-------- ");
            total[j] += time;
        }
        printf("\n");
    }
    printf("\nTotal (lost) time:\n%s\n", title);
    for (j=0; j < rj; j++) {
        printf("%f ", total[j]);
    }
    printf("\n");

    return 0;
}
