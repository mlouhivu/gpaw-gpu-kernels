#ifndef KERNEL_H_
#define KERNEL_H_

#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"

float run_kernel(double *x_, const int3 sizex, const int3 pos,
                 double *y_, const int3 sizey,
                 const unsigned int layers,
                 dim3 blx, dim3 thx, char *title, char *header);

#endif
