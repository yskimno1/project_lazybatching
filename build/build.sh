#/usr/bin/bash
/usr/local/cuda-10.1/bin/nvcc -c $HOME/new/Project/model.cu  -lcudnn -lcublas -lz -std=c++11
/usr/local/cuda-10.1/bin/nvcc -c $HOME/new/Project/layer.cu  -lcudnn -lcublas -lz -std=c++11
/usr/local/cuda-10.1/bin/nvcc -o $HOME/new/Project/anybatch $HOME/new/Project/cudnn.cu $HOME/new/Project/build/layer.o $HOME/new/Project/build/model.o -lcudnn -lcublas -lz -std=c++11
