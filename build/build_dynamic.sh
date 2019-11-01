#/usr/bin/bash

PROJECT_PATH=$HOME/XBatch/XBatch/Project

/usr/local/cuda-10.1/bin/nvcc -c $PROJECT_PATH/model.o $PROJECT_PATH/model.cu  -lcudnn -lcublas -lz -std=c++11
/usr/local/cuda-10.1/bin/nvcc -c $PROJECT_PATH/layer.o $PROJECT_PATH/layer.cu  -lcudnn -lcublas -lz -std=c++11
/usr/local/cuda-10.1/bin/nvcc -o $PROJECT_PATH/dynamic $PROJECT_PATH/dynamic.cu $PROJECT_PATH/layer.o $PROJECT_PATH/model.o -lcudnn -lcublas -lz -std=c++11
