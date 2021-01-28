#ifndef KERNELS_H_
#define KERNELS_H_

#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"

float run_kernel0(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey, const int layers,
                  char *title, char *header,
                  const int repeat, const int trial);

float run_kernel1(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey, const int layers,
                  char *title, char *header,
                  const int repeat, const int trial);

float run_kernel2(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey, const int layers,
                  char *title, char *header,
                  const int repeat, const int trial);

float run_kernel2b(double *x_, const int3 sizex, const int3 pos,
                   double *y_, const int3 sizey, const int layers,
                   char *title, char *header,
                   const int repeat, const int trial);

#endif
