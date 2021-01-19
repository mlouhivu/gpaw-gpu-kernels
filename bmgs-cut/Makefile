CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

objects := $(patsubst %.cu, %.o, $(wildcard *.cu))

bmgs-cut: $(objects)
	$(CC) -o $@ $(objects)

%.o: %.cu
	$(CC) -c $(CFLAGS) $< -o $@
