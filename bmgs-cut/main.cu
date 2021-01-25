#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"
#include "config.h"
#include "kernels.h"


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

void reset(double *x, double *x_, int n,
           double *y, double *y_, int m, int layers)
{
    int i;

    for (i=0; i < layers * n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);
}

int run(const unsigned int layers, const int3 sizex, const int3 sizey,
        int3 pos, float *results, char *title,
        const int repeat, const int trials)
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

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ri=0;

    kernel_func kp[MAX_KERNELS];
    get_kernels(kp);

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
    cudaEventRecord(start);
    for (i=0; i < repeat; i++) {
        for (l=0; l < layers; l++) {
            bmgs_cut(xp + l * n, dimx, position, yp + l * m, dimy);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "CPU");
    sprintf(title, "%8s", name);
    check_result(name, &y_ref[0], yp, layers * m, time, verbose);
    results[ri++] = time / repeat;

    /*** GPU implementations ***/
    i = 0;
    while (kp[i] != NULL) {
        reset(x, x_, n, y, y_, m, layers);

        // launch kernel
        time = 0.0;
        for (j=0; j < trials; j++) {
            time += kp[i](x_, sizex, pos, y_, sizey, layers, title, header,
                          repeat, j);
        }
        time /= trials;
        i++;

        cudaMemcpy(&y, y_, sizeof(double) * m * layers, cudaMemcpyDeviceToHost);
        check_result(header, &y_ref[0], yp, layers * m, time, verbose);
        results[ri++] = time / repeat;
    }

    return ri;
}

void parse_args(int argc, char *argv[], int *repeat, int *trials)
{
    switch (argc) {
        case 1:
            break;
        case 2:
            // one argument
            (*repeat) = atoi(argv[1]);
            break;
        case 3:
            // two arguments
            (*repeat) = atoi(argv[1]);
            (*trials) = atoi(argv[2]);
            break;
        default:
            printf("Usage: bmgs-cut {repeat} {trials}\n");
            exit(-1);
    }
}


int main(int argc, char *argv[])
{
    int i, j;

    int repeat = DEFAULT_REPEAT;
    int trials = DEFAULT_TRIALS;

    int ri = 0;
    int rj = 0;
    float results[MAX_ARGS][MAX_KERNELS];
    float best[MAX_ARGS];
    float total[MAX_KERNELS];
    float time;
    float best_total = 9999.0;

    char title[512];

    t_config config;
    t_arg arg;

    parse_args(argc, argv, &repeat, &trials);

    for (i=0; i < MAX_KERNELS; i++)
        total[i] = 0.0;

    printf("# BMGS-CUT\n");
    printf("#  measurements:    %d\n", trials);
    printf("#  kernel launches: %d\n", repeat);

    config = get_config();
    for (i=0; i < config.nargs; i++) {
        arg = config.args[i];
        j = run(arg.layers, arg.dimx, arg.dimy, arg.position,
                results[ri++], title, repeat, trials);
        rj = MAX(rj, j);
    }

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
        best_total = MIN(best_total, total[j]);
    }
    printf("  (absolute)\n");
    for (j=0; j < rj; j++) {
        time = total[j] - best_total;
        if (time > 0.0)
            printf("%f ", time);
        else
            printf("-------- ");
    }
    printf("  (relative difference)\n");

    return 0;
}
