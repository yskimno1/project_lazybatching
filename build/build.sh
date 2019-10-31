#/usr/bin/bash
/usr/local/cuda-10.1/bin/nvcc -c $HOME/new/new_new/Project/model.o $HOME/new/new_new/Project/model.cu  -lcudnn -lcublas -lz -std=c++11
/usr/local/cuda-10.1/bin/nvcc -c $HOME/new/new_new/Project/layer.o $HOME/new/new_new/Project/layer.cu  -lcudnn -lcublas -lz -std=c++11
/usr/local/cuda-10.1/bin/nvcc -o $HOME/new/new_new/Project/anybatch $HOME/new/new_new/Project/cudnn.cu $HOME/new/new_new/Project/layer.o $HOME/new/new_new/Project/model.o -lcudnn -lcublas -lz -std=c++11
