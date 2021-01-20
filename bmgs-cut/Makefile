CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

objects := $(patsubst %.cu, %.o, $(wildcard *.cu))

bmgs-cut: $(objects)
	$(CC) -o $@ $(objects)

main.o: main.cu common.h
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cu %.h common.h
	$(CC) -c $(CFLAGS) $< -o $@
