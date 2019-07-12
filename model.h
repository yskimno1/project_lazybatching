// #include <stdio.h>
// #include <string.h>

// #include <sstream>
// #include <fstream>
// #include <stdlib.h>
// #include <iostream>
// #include <assert.h>

// #include <iostream>
// #include <vector>

// #include <cudnn.h>
// #include <cublas_v2.h>
// // #include <cnmem/cnmem.h>

// #include <unistd.h>
// #include <time.h>
// #include <pthread.h>

// #include "layer.h"

struct Model_layer{
    Layer* list_layer[MAX_LAYER_NUM];

    bool data_exist;
};

struct Model_layer createModel(cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem, 
                    int n_in, int c_in, int h_in, int w_in, int pad_h, int pad_w, int stride_h, int stride_w, 
                    int k, int r, int s, int n_out, int c_out, int h_out, int w_out);