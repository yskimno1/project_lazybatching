#/usr/bin/bash
/usr/local/cuda-8.0/bin/nvcc -o dynamic dynamic.cu -lcudnn -lcublas -lz
