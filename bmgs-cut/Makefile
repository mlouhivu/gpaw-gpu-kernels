CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

exe := bmgs-cut
objects := $(patsubst %.cu, %.o, $(wildcard *.cu))

$(exe): $(objects)
	$(CC) -o $@ $(objects)

main.o: main.cu common.h
	$(CC) -c $(CFLAGS) $< -o $@

config.o: config.cu config.h common.h
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cu kernels.h common.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm *.o $(exe)
