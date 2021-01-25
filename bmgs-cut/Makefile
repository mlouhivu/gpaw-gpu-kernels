CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

objects := $(patsubst %.cu, %.o, $(wildcard *.cu))

bmgs-cut: $(objects)
	$(CC) -o $@ $(objects)

main.o: main.cu common.h
	$(CC) -c $(CFLAGS) $< -o $@

config.o: config.cu config.h common.h
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cu kernels.h common.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm *.o bmgs-cut
