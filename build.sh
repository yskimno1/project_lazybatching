#/usr/bin/bash
/usr/local/cuda-8.0/bin/nvcc -o anybatch cudnn.cu -lcudnn -lcublas -lz 
