#ifndef CONFIG_H_
#define CONFIG_H_

#include <cuda_runtime.h>
#include "kernels.h"

#define MAX_ARGS 512

typedef struct {
    int layers;
    int3 dimx;
    int3 dimy;
    int3 position;
} t_arg;

typedef struct {
    int nargs;
    t_arg args[MAX_ARGS];
} t_config;

t_arg as_arg(int layers, int3 dimx, int3 dimy, int3 position);
t_config get_config();
void get_kernels(kernel_func *kernels);

#endif
