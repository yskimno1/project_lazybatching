#/usr/bin/bash
/usr/local/cuda-8.0/bin/nvcc -c $HOME/Project/model.cu  -lcudnn -lcublas -lz
/usr/local/cuda-8.0/bin/nvcc -c $HOME/Project/layer.cu  -lcudnn -lcublas -lz
/usr/local/cuda-8.0/bin/nvcc -o $HOME/Project/dynamic $HOME/Project/dynamic.cu $HOME/Project/build/layer.o $HOME/Project/build/model.o -lcudnn -lcublas -lz
