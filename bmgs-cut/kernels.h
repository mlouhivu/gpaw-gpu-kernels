#ifndef KERNELS_H_
#define KERNELS_H_

#include <stdio.h>
#include "common.h"

float run_kernel0(Tcuda *x_, const int3 sizex, const int3 pos,
                  Tcuda *y_, const int3 sizey, const int layers,
                  const Tcuda phase_,
                  char *title, char *header,
                  const int repeat, const int trial);

float run_kernel1(Tcuda *x_, const int3 sizex, const int3 pos,
                  Tcuda *y_, const int3 sizey, const int layers,
                  const Tcuda phase_,
                  char *title, char *header,
                  const int repeat, const int trial);

float run_kernel2(Tcuda *x_, const int3 sizex, const int3 pos,
                  Tcuda *y_, const int3 sizey, const int layers,
                  const Tcuda phase_,
                  char *title, char *header,
                  const int repeat, const int trial);

float run_kernel2b(Tcuda *x_, const int3 sizex, const int3 pos,
                   Tcuda *y_, const int3 sizey, const int layers,
                   const Tcuda phase_,
                   char *title, char *header,
                   const int repeat, const int trial);

float run_kernel3(Tcuda *x_, const int3 sizex, const int3 pos,
                  Tcuda *y_, const int3 sizey, const int layers,
                  const Tcuda phase_,
                  char *title, char *header,
                  const int repeat, const int trial);

float run_kernel3b(Tcuda *x_, const int3 sizex, const int3 pos,
                   Tcuda *y_, const int3 sizey, const int layers,
                   const Tcuda phase_,
                   char *title, char *header,
                   const int repeat, const int trial);

float run_kernel3c(Tcuda *x_, const int3 sizex, const int3 pos,
                   Tcuda *y_, const int3 sizey, const int layers,
                   const Tcuda phase_,
                   char *title, char *header,
                   const int repeat, const int trial);

float run_kernel4(Tcuda *x_, const int3 sizex, const int3 pos,
                  Tcuda *y_, const int3 sizey, const int layers,
                  const Tcuda phase_,
                  char *title, char *header,
                  const int repeat, const int trial);

float run_kernel4b(Tcuda *x_, const int3 sizex, const int3 pos,
                   Tcuda *y_, const int3 sizey, const int layers,
                   const Tcuda phase_,
                   char *title, char *header,
                   const int repeat, const int trial);

float run_kernel5(Tcuda *x_, const int3 sizex, const int3 pos,
                  Tcuda *y_, const int3 sizey, const int layers,
                  const Tcuda phase_,
                  char *title, char *header,
                  const int repeat, const int trial);

#endif
