CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

exe := bmgs-cut
objects := $(patsubst %.cu, %.o, $(wildcard *.cu))

exez := bmgs-cutz
objectz := $(patsubst %.cu, %.z.o, $(wildcard *.cu))

all: $(exe) $(exez)

$(exe): $(objects)
	$(CC) -o $@ $(objects)

$(exez): $(objectz)
	$(CC) -DUSE_COMPLEX -o $@ $(objectz)

main.o: main.cu common.h
	$(CC) -c $(CFLAGS) $< -o $@

config.o: config.cu config.h common.h
	$(CC) -c $(CFLAGS) $< -o $@

main.z.o: main.cu common.h
	$(CC) -DUSE_COMPLEX -c $(CFLAGS) $< -o $@

config.z.o: config.cu config.h common.h
	$(CC) -DUSE_COMPLEX -c $(CFLAGS) $< -o $@

%.z.o: %.cu kernels.h common.h
	$(CC) -DUSE_COMPLEX -c $(CFLAGS) $< -o $@

%.o: %.cu kernels.h common.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm *.o $(exe) $(exez)
