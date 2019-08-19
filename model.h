#include <stdio.h>
#include <string.h>

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <iostream>
#include <vector>

#include <cudnn.h>
#include <cublas_v2.h>
// #include <cnmem/cnmem.h>

#include <unistd.h>
#include <time.h>
#include <pthread.h>

#ifndef LAYER
#include "layer.h"
#define LAYER
#endif

struct Model_layer{
    Layer* list_layer[MAX_LAYER_NUM];

    int data_first;
    int data_num;
    bool data_exist;
    void* matrix_W1;
    void* matrix_W2;
    void* vector_score;


};

struct Model_layer create_GNMT(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem);
struct Model_layer create_Resnet(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem);
