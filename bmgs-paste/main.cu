#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"
#include "config.h"
#include "kernels.h"


void bmgs_paste(const double *a, const int sizea[3],
                double *b, const int sizeb[3], const int startb[3])
{
  b += startb[2] + (startb[1] + startb[0] * sizeb[1]) * sizeb[2];
  for (int i0 = 0; i0 < sizea[0]; i0++)
    {
      for (int i1 = 0; i1 < sizea[1]; i1++)
        {
          memcpy(b, a, sizea[2] * sizeof(double));
          a += sizea[2];
          b += sizeb[2];
        }
      b += sizeb[2] * (sizeb[1] - sizea[1]);
    }
}

void reference(double *y_ref, double *x, const int layers,
               const int3 sizex, const int3 sizey, const int3 pos)
{
    int i, j, k, l, s, t;

    for (l=0; l < layers; l++) {
        for (i=0; i < sizex.x; i++) {
            for (j=0; j < sizex.y; j++) {
                for (k=0; k < sizex.z; k++) {
                    t = sizey.z * sizey.y * sizey.x * l
                      + sizey.z * sizey.y * (i + pos.x)
                      + sizey.z * (j + pos.y)
                      + k + pos.z;
                    s = sizex.z * sizex.y * sizex.x * l
                      + sizex.z * sizex.y * i
                      + sizex.z * j
                      + k;
                    y_ref[t] = x[s];
                }
            }
        }
    }
}

double variance(double *reference, double *result, const int layers,
                const int3 size, const int3 slice, const int3 position)
{
    int i, j, k, l, index;
    double error = 0.0;
    double diff;
    int n = layers * slice.x * slice.y * slice.z;

    for (l=0; l < layers; l++) {
        for (i=0; i < slice.x; i++) {
            for (j=0; j < slice.y; j++) {
                for (k=0; k < slice.z; k++) {
                    index = size.z * size.y * size.x * l
                          + size.z * size.y * (i + position.x)
                          + size.z * (j + position.y)
                          + k + position.z;
                    diff = reference[index] - result[index];
                    error += diff * diff;
                }
            }
        }
    }
    return sqrt(error) / n;
}

void echo(const char *title, double *x, const int3 size, const int layers)
{
    printf("%s\n", title);
    for (int l=0; l < layers; l++) {
        for (int k=0; k < size.z; k++) {
            for (int i=0; i < size.x; i++) {
                for (int j=0; j < size.y; j++) {
                    int index = size.z * size.y * size.x * l
                          + size.z * size.y * i
                          + size.z * j
                          + k;
                    printf(" %f", x[index]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void check_result(const char *name, double *y_ref, double *y,
                  const int layers, const int3 slice, const int3 size,
                  const int3 position, double time, int verbose)
{
    double error;
    const int start = size.z * size.y * position.x
                    + size.z * position.y
                    + position.z;
    const int end = size.z * size.y * size.x * (layers - 1)
                  + size.z * size.y * (position.x + slice.x)
                  + size.z * (position.y + slice.y)
                  + position.z + slice.z;

    error = variance(y_ref, y, layers, size, slice, position);
    if (error || verbose) {
        printf("\n%s\n", name);
        printf(" position: (%d,%d,%d)\n", position.x, position.y, position.z);
        printf("    slice: (%d,%d,%d)\n", slice.x, slice.y, slice.z);
        printf("reference: %f %f %f ... %f %f\n",
                y_ref[start + 0], y_ref[start + 1], y_ref[start + 2],
                y_ref[end - 2], y_ref[end - 1]);
        printf("   result: %f %f %f ... %f %f\n",
                y[start + 0], y[start + 1], y[start + 2],
                y[end - 2], y[end - 1]);
        printf(" variance: %f\n", error);
        printf("exec time: %f\n", time);
        if (verbose > 1)
            echo("Y", y, size, layers);
    }
}

void reset(double *x, double *x_, int n,
           double *y, double *y_, int m, int layers)
{
    int i;

    for (i=0; i < layers * n; i++) {
        x[i] = 1.0 + (double) i / 1000.0;
    }
    for (i=0; i < layers * m; i++) {
        y[i] = 0.0;
    }
    cudaMemcpy(x_, x, sizeof(double) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m * layers, cudaMemcpyHostToDevice);
}

int run(const int layers, const int3 sizex, const int3 sizey,
        int3 pos, float *results, char *title,
        const int repeat, const int trials)
{
    int i, j, l;
    int verbose = 0;
    int debug = (verbose > 1) ? 1: 0;
    char header[512];
    char name[32];

    const int dimx[3] = {sizex.x, sizex.y, sizex.z};
    const int dimy[3] = {sizey.x, sizey.y, sizey.z};
    const int position[3] = {pos.x, pos.y, pos.z};

    const int n = sizex.x * sizex.y * sizex.z;
    const int m = sizey.x * sizey.y * sizey.z;
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

    // allocate GPU arrays
    cudaMalloc((void **) &x_, sizeof(double) * n * layers);
    cudaMalloc((void **) &y_, sizeof(double) * m * layers);

    // initialise data
    reset(x, x_, n, y, y_, m, layers);
    if (debug)
        echo("X", x, sizex, layers);

    // get reference values
    reference(&y_ref[0], xp, layers, sizex, sizey, pos);
    if (debug)
        echo("Y_REF", y_ref, sizey, layers);

    /*** CPU implementation ***/
    cudaEventRecord(start);
    for (i=0; i < repeat; i++) {
        for (l=0; l < layers; l++) {
            bmgs_paste(xp + l * n, dimx, yp + l * m, dimy, position);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sprintf(name, "CPU");
    sprintf(title, "%8s", name);
    check_result(name, &y_ref[0], yp, layers, sizex, sizey, pos, time, verbose);
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
        check_result(header, &y_ref[0], yp, layers, sizex, sizey, pos, time,
                     verbose);
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
            printf("Usage: bmgs-paste {repeat} {trials}\n");
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

    printf("# BMGS-PASTE\n");
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
