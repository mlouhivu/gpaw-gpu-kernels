#ifndef KERNELS_H_
#define KERNELS_H_

#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"

float run_kernel0(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey,
                  const unsigned int layers,
                  char *title, char *header);

float run_kernel1(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey,
                  const unsigned int layers,
                  char *title, char *header);

float run_kernel2(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey,
                  const unsigned int layers,
                  char *title, char *header);

float run_kernel2b(double *x_, const int3 sizex, const int3 pos,
                   double *y_, const int3 sizey,
                   const unsigned int layers,
                   char *title, char *header);

float run_kernel3(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey,
                  const unsigned int layers,
                  char *title, char *header);

float run_kernel3b(double *x_, const int3 sizex, const int3 pos,
                   double *y_, const int3 sizey,
                   const unsigned int layers,
                   char *title, char *header);

float run_kernel3c(double *x_, const int3 sizex, const int3 pos,
                   double *y_, const int3 sizey,
                   const unsigned int layers,
                   char *title, char *header);

float run_kernel4(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey,
                  const unsigned int layers,
                  char *title, char *header);

float run_kernel4b(double *x_, const int3 sizex, const int3 pos,
                   double *y_, const int3 sizey,
                   const unsigned int layers,
                   char *title, char *header);

float run_kernel5(double *x_, const int3 sizex, const int3 pos,
                  double *y_, const int3 sizey,
                  const unsigned int layers,
                  char *title, char *header);

#endif
