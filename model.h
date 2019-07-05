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

enum model_mode{
    DEFAULT_STATE   = 0,
    GLOBAL          = 1,
    LOCAL           = 2
};

struct Model_layer{
    Layer* list_layer[MAX_LAYER_NUM];
    enum model_mode current_mode;
    int current_index;
    int global_index;
    int data_exist;
};

struct Model_layer createModel(cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem, 
                    int n_in, int c_in, int h_in, int w_in, int pad_h, int pad_w, int stride_h, int stride_w, 
                    int k, int r, int s, int n_out, int c_out, int h_out, int w_out);