#include "config.h"

t_arg as_arg(int layers, int3 dimx, int3 dimy, int3 position)
{
    t_arg arg;

    arg.layers = layers;
    arg.dimx = dimx;
    arg.dimy = dimy;
    arg.position = position;

    return arg;
}

t_config get_config()
{
    t_config config;
    config.nargs = 0;

    int layers;
    int3 dimx;
    int3 dimy;
    int3 position;

    // carbon nanotube
    layers = 56;
    dimx = {41,21,32};
    dimy = {41,21,1};
    position = {0,0,0};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 56;
    dimx = {85,46,68};
    dimy = {79,40,3};
    position = {3,3,62};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 56;
    dimx = {85,45,68};
    dimy = {79,39,3};
    position = {3,3,3};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 56;
    dimx = {21,11,17};
    dimy = {19,1,15};
    position = {1,9,1};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 56;
    dimx = {21,11,18};
    dimy = {19,9,1};
    position = {1,1,1};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    // copper filament
    layers = 25;
    dimx = {89,52,62};
    dimy = {83,46,3};
    position = {3,3,56};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 25;
    dimx = {43,24,29};
    dimy = {43,24,1};
    position = {0,0,0};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 25;
    dimx = {43,25,30};
    dimy = {41,24,28};
    position = {1,1,1};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 48;
    dimx = {89,52,62};
    dimy = {83,46,3};
    position = {3,3,56};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    // single fullerene
    layers = 1;
    dimx = {6,7,12};
    dimy = {1,5,11};
    position = {0,1,0};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 1;
    dimx = {12,12,23};
    dimy = {1,11,22};
    position = {0,0,0};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    // other
    layers = 12;
    dimx = {252,31,64};
    dimy = {252,31,1};
    position = {0,0,0};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    layers = 8;
    dimx = {100,100,100};
    dimy = {10,10,10};
    position = {22,44,66};
    config.args[config.nargs++] = as_arg(layers, dimx, dimy, position);

    return config;
}

void get_kernels(kernel_func *kernels)
{
    for (int j=0; j < MAX_KERNELS; j++)
        kernels[j] = NULL;

    int i = 0;
    kernels[i++] = &run_kernel0;
    kernels[i++] = &run_kernel1;
    kernels[i++] = &run_kernel2;
    kernels[i++] = &run_kernel2b;
    kernels[i++] = &run_kernel3;
    kernels[i++] = &run_kernel3b;
    kernels[i++] = &run_kernel3c;
    kernels[i++] = &run_kernel4;
    kernels[i++] = &run_kernel4b;
    kernels[i++] = &run_kernel5;
}
