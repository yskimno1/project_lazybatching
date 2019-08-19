#/usr/bin/bash
/usr/local/cuda-8.0/bin/nvcc -c $HOME/Project/model.cu  -lcudnn -lcublas -lz
/usr/local/cuda-8.0/bin/nvcc -c $HOME/Project/layer.cu  -lcudnn -lcublas -lz
/usr/local/cuda-8.0/bin/nvcc -o $HOME/Project/anybatch $HOME/Project/cudnn.cu $HOME/Project/build/layer.o $HOME/Project/build/model.o -lcudnn -lcublas -lz 
