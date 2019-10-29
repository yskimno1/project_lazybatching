#/usr/bin/bash
/usr/local/cuda-10.1/bin/nvcc -c $HOME/Project/model.cu  -lcudnn -lcublas -lz -std=c++11
/usr/local/cuda-10.1/bin/nvcc -c $HOME/Project/layer.cu  -lcudnn -lcublas -lz -std=c++11
/usr/local/cuda-10.1/bin/nvcc -o $HOME/Project/dynamic $HOME/Project/dynamic.cu $HOME/Project/build/layer.o $HOME/Project/build/model.o -lcudnn -lcublas -lz -std=c++11
