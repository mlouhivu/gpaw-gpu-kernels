#include <stdio.h>
#include "common.h"
#include "config.h"
#include "kernels.h"

extern const T default_phase;
extern const Tcuda default_phase_;

void bmgs_cut(const T *a, const int n[3], const int c[3],
              T *b, const int m[3])
{
  a += c[2] + (c[1] + c[0] * n[1]) * n[2];
  for (int i0 = 0; i0 < m[0]; i0++)
    {
      for (int i1 = 0; i1 < m[1]; i1++)
        {
          memcpy(b, a, m[2] * sizeof(T));
          a += n[2];
          b += m[2];
        }
      a += n[2] * (n[1] - m[1]);
    }
}

void bmgs_cutmz(const T* a, const int sizea[3], const int start[3],
                T* b, const int sizeb[3], T p)
{
    a += start[2] + (start[1] + start[0] * sizea[1]) * sizea[2];
    for (int i0 = 0; i0 < sizeb[0]; i0++)
    {
        for (int i1 = 0; i1 < sizeb[1]; i1++)
        {
            for (int i2 = 0; i2 < sizeb[2]; i2++)
                b[i2] = p * a[i2];
            a += sizea[2];
            b += sizeb[2];
        }
        a += sizea[2] * (sizea[1] - sizeb[1]);
    }
}

void reference(T *y_ref, T *x, const int layers,
               const int3 sizex, const int3 sizey, const int3 pos, T phase)
{
    int i, j, k, l, s, t;

    for (l=0; l < layers; l++) {
        for (i=0; i < sizey.x; i++) {
            for (j=0; j < sizey.y; j++) {
                for (k=0; k < sizey.z; k++) {
                    t = sizey.z * sizey.y * sizey.x * l
                      + sizey.z * sizey.y * i
                      + sizey.z * j
                      + k;
                    s = sizex.z * sizex.y * sizex.x * l
                      + sizex.z * sizex.y * (i + pos.x)
                      + sizex.z * (j + pos.y)
                      + k + pos.z;
                    y_ref[t] = phase * x[s];
                }
            }
        }
    }
}

double variance(T *reference, T *result, int n)
{
    int i;
    double error = 0.0;
    double diff;

    for (i=0; i < n; i++) {
        diff = ABS(reference[i] - result[i]);
        error += diff * diff;
    }
    return sqrt(error) / n;
}

void check_result(const char *name, T *y_ref, T *y, int n,
                  double time, int verbose)
{
    double error;

    error = variance(y_ref, y, n);
    if (error || verbose) {
        printf("\n%s\n", name);
#ifndef USE_COMPLEX
        printf("reference: %f %f %f ... %f %f\n",
                y_ref[0], y_ref[1], y_ref[2], y_ref[n - 2], y_ref[n - 1]);
        printf("   result: %f %f %f ... %f %f\n",
                y[0], y[1], y[2], y[n - 2], y[n - 1]);
#else
        printf("reference: (%f,%f) (%f,%f) ... (%f,%f)\n",
                y_ref[0].x, y_ref[0].y, y_ref[1].x, y_ref[1].y,
                y_ref[n - 1].x, y_ref[n - 1].y);
        printf("   result: (%f,%f) (%f,%f) ... (%f,%f)\n",
                y[0].x, y[0].y, y[1].x, y[1].y, y[n - 1].x, y[n - 1].y);
#endif
        printf(" variance: %f\n", error);
        printf("exec time: %f\n", time);
    }
}

void reset(T *x, Tcuda *x_, int n,
           T *y, Tcuda *y_, int m, int layers)
{
    int i;

    for (i=0; i < layers * n; i++) {
#ifdef USE_COMPLEX
        x[i].x = 1.0 + (double) i / 10000.0;
        x[i].y = 1.0 - (double) i / 10000.0;
#else
        x[i] = 1.0 + (double) i / 10000.0;
#endif
    }
    for (i=0; i < layers * m; i++) {
#ifdef USE_COMPLEX
        y[i].x = 0.0;
        y[i].y = 0.0;
#else
        y[i] = 0.0;
#endif
    }
    cudaMemcpy(x_, x, sizeof(T) * n * layers, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(T) * m * layers, cudaMemcpyHostToDevice);
}

int run(const int layers, const int3 sizex, const int3 sizey,
        int3 pos, float *results, char *title,
        const int repeat, const int trials)
{
    int i, j, l;
    int verbose = 0;
    char header[512];
    char name[32];

    const int dimx[3] = {sizex.x, sizex.y, sizex.z};
    const int dimy[3] = {sizey.x, sizey.y, sizey.z};
    const int position[3] = {pos.x, pos.y, pos.z};

    const int n = sizex.x * sizex.y * sizex.z;
    const int m = sizey.x * sizey.y * sizey.z;
    T x[layers * n], y[layers * m], y_ref[layers * m];
    T *xp = &x[0];
    T *yp = &y[0];
    Tcuda *x_, *y_;

#ifdef USE_COMPLEX
    const T phase = AS_T(0.5, -0.5);
    const Tcuda phase_ = AS_TCUDA(0.5, -0.5);
#else
    const T phase = 1.0;
    const Tcuda phase_ = 1.0;
#endif

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ri=0;

    kernel_func kp[MAX_KERNELS];
    get_kernels(kp);

    // allocate GPU arrays
    cudaMalloc((void **) &x_, sizeof(Tcuda) * n * layers);
    cudaMalloc((void **) &y_, sizeof(Tcuda) * m * layers);

    // initialise data
    reset(x, x_, n, y, y_, m, layers);

    // get reference values
    reference(&y_ref[0], xp, layers, sizex, sizey, pos, phase);

    /*** CPU implementation ***/
    cudaEventRecord(start);
    for (i=0; i < repeat; i++) {
        for (l=0; l < layers; l++) {
#ifdef USE_COMPLEX
            bmgs_cutmz(xp + l * n, dimx, position, yp + l * m, dimy, phase);
#else
            bmgs_cut(xp + l * n, dimx, position, yp + l * m, dimy);
#endif
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
            time += kp[i](x_, sizex, pos, y_, sizey, layers, phase_,
                          title, header, repeat, j);
        }
        time /= trials;
        i++;

        cudaMemcpy(&y, y_, sizeof(T) * m * layers, cudaMemcpyDeviceToHost);
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

#ifdef USE_COMPLEX
    printf("# BMGS-CUTZ\n");
#else
    printf("# BMGS-CUT\n");
#endif
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
