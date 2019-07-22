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

    int data_first;
    int data_num;
    bool data_exist;
};

struct Model_layer createModel(int type, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem);