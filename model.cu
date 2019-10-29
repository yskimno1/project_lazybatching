#include <stdio.h>
#include <string.h>

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <assert.h>

#include <ctype.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#include <cudnn.h>
#include <cublas_v2.h>

#include <unistd.h>
#include <time.h>
#include <pthread.h>

#ifdef USE_CPP_11
#include <thread>
#endif

#include "model.h"

#ifndef LAYER
#include "layer.h"
#define LAYER
#endif

struct Model_layer create_GNMT(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem){
    struct Model_layer model;
    printf("create GNMT\n");

    int seq_length_GNMT = 1;
    int num_layers_GNMT = 1;

    void* matrix_W1;
    void* matrix_W2;
    void* vector_weight;

    checkCuda(cudaMalloc(&matrix_W1, HIDDEN_SIZE_GNMT*HIDDEN_SIZE_GNMT*sizeof(value_type)));
    checkCuda(cudaMalloc(&matrix_W2, HIDDEN_SIZE_GNMT*HIDDEN_SIZE_GNMT*sizeof(value_type)));
    checkCuda(cudaMalloc(&vector_weight, HIDDEN_SIZE_GNMT*sizeof(value_type)));

    model.matrix_W1 = matrix_W1;
    model.matrix_W2 = matrix_W2;
    model.vector_score = vector_weight;

    model.list_layer[0] = new Layer(RNN, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        HIDDEN_SIZE_GNMT, HIDDEN_SIZE_GNMT, num_layers_GNMT, 0);

    model.list_layer[1] = new Layer(RNN, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        HIDDEN_SIZE_GNMT*2, HIDDEN_SIZE_GNMT, num_layers_GNMT, 1);
    model.list_layer[0]->setDstData(model.list_layer[1]->SrcData());

    model.list_layer[2] = new Layer(RNN, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        HIDDEN_SIZE_GNMT, HIDDEN_SIZE_GNMT, num_layers_GNMT, 2);
    model.list_layer[1]->setDstData(model.list_layer[2]->SrcData());

    model.list_layer[3] = new Layer(RNN, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        HIDDEN_SIZE_GNMT, HIDDEN_SIZE_GNMT, num_layers_GNMT, 3);
    model.list_layer[2]->setDstData(model.list_layer[3]->SrcData());

    model.list_layer[4] = new Layer(ATTENTION, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        HIDDEN_SIZE_GNMT, HIDDEN_SIZE_GNMT, num_layers_GNMT, 4);
    model.list_layer[3]->setDstData(model.list_layer[4]->SrcData());

    /* decoder */

    model.list_layer[5] = new Layer(RNN_DECODER, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        1, HIDDEN_SIZE_GNMT, num_layers_GNMT, 5);
        
    // model.list_layer[4]->setContextData(model.list_layer[5]->SrcData());

    model.list_layer[6] = new Layer(RNN_DECODER, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        1, HIDDEN_SIZE_GNMT, num_layers_GNMT, 6);

    model.list_layer[5]->setDstData(model.list_layer[6]->SrcData());

    model.list_layer[7] = new Layer(RNN_DECODER, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        1, HIDDEN_SIZE_GNMT, num_layers_GNMT, 7);

    model.list_layer[6]->setDstData(model.list_layer[7]->SrcData());

    model.list_layer[8] = new Layer(RNN_LAST, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        1, HIDDEN_SIZE_GNMT, num_layers_GNMT, 8);

    model.list_layer[7]->setDstData(model.list_layer[8]->SrcData());


    return model;
}

struct Model_layer create_Resnet(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem){
    struct Model_layer model;
    model.list_layer[0] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 3, 224, 224,
        3, 3, 2, 2,
        64, 7, 7,
        MAX_BATCH_SIZE, 64, 112, 112,
        1, 20, 2000, 0);

    model.list_layer[1] = new Layer(POOL, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 64, 112, 112,
        0, 0, 2, 2,
        64, 3, 3,
        MAX_BATCH_SIZE, 64, 56, 56,
        1, 20, 2000, 1);
    model.list_layer[0]->setDstData(model.list_layer[1]->SrcData());

    /* loop */
    
    model.list_layer[2] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 64, 56, 56,
        0, 0, 1, 1,
        64, 1, 1,
        MAX_BATCH_SIZE, 64, 56, 56,
        1, 20, 2000, 2);

    model.list_layer[1]->setDstData(model.list_layer[2]->SrcData());

    model.list_layer[3] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 64, 56, 56,
        1, 1, 1, 1,
        64, 3, 3,
        MAX_BATCH_SIZE, 64, 56, 56,
        1, 20, 2000, 3);

    model.list_layer[2]->setDstData(model.list_layer[3]->SrcData());

    model.list_layer[4] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 64, 56, 56,
        0, 0, 1, 1,
        256, 1, 1,
        MAX_BATCH_SIZE, 256, 56, 56,
        1, 20, 2000, 4);

    model.list_layer[3]->setDstData(model.list_layer[4]->SrcData());

    for(int k=1; k<3; k++){

        model.list_layer[3*k+2] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 256, 56, 56,
            0, 0, 1, 1,
            64, 1, 1,
            MAX_BATCH_SIZE, 64, 56, 56,
            1, 20, 2000, 3*k+2);

        model.list_layer[3*k+1]->setDstData(model.list_layer[3*k+2]->SrcData());

        model.list_layer[3*k+3] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 64, 56, 56,
            1, 1, 1, 1,
            64, 3, 3,
            MAX_BATCH_SIZE, 64, 56, 56,
            1, 20, 2000, 3*k+3);

        model.list_layer[3*k+2]->setDstData(model.list_layer[3*k+3]->SrcData());

        model.list_layer[3*k+4] = new Layer(CONV_RESIDUAL, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 64, 56, 56,
            0, 0, 1, 1,
            256, 1, 1,
            MAX_BATCH_SIZE, 256, 56, 56,
            1, 20, 2000, 3*k+4);

        model.list_layer[3*k+3]->setDstData(model.list_layer[3*k+4]->SrcData());
    }

    /* second loop 1 */

    model.list_layer[11] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 256, 56, 56,
        0, 0, 2, 2,
        128, 1, 1,
        MAX_BATCH_SIZE, 128, 28, 28,
        1, 20, 2000, 11);
    model.list_layer[10]->setDstData(model.list_layer[11]->SrcData());

    model.list_layer[12] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 128, 28, 28,
        1, 1, 1, 1,
        128, 3, 3,
        MAX_BATCH_SIZE, 128, 28, 28,
        1, 20, 2000, 12);
    model.list_layer[11]->setDstData(model.list_layer[12]->SrcData());

    model.list_layer[13] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 128, 28, 28,
        0, 0, 1, 1,
        512, 1, 1,
        MAX_BATCH_SIZE, 512, 28, 28,
        1, 20, 2000, 13);        
    model.list_layer[12]->setDstData(model.list_layer[13]->SrcData());

    for(int k=1; k<4; k++){
        model.list_layer[3*k+11] = new Layer(CONV, &cublasHandle,  &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 512, 28, 28,
            0, 0, 1, 1,
            128, 1, 1,
            MAX_BATCH_SIZE, 128, 28, 28,
            1, 20, 2000, 3*k+11);
        model.list_layer[3*k+10]->setDstData(model.list_layer[3*k+11]->SrcData());

        model.list_layer[3*k+12] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 128, 28, 28,
            1, 1, 1, 1,
            128, 3, 3,
            MAX_BATCH_SIZE, 128, 28, 28,
            1, 20, 2000, 3*k+12);
        model.list_layer[3*k+11]->setDstData(model.list_layer[3*k+12]->SrcData());

        model.list_layer[3*k+13] = new Layer(CONV_RESIDUAL, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 128, 28, 28,
            0, 0, 1, 1,
            512, 1, 1,
            MAX_BATCH_SIZE, 512, 28, 28,
            1, 20, 2000, 3*k+13);        
        model.list_layer[3*k+12]->setDstData(model.list_layer[3*k+13]->SrcData());
    }

    /* third loop */
    model.list_layer[23] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 512, 28, 28,
        0, 0, 2, 2,
        256, 1, 1,
        MAX_BATCH_SIZE, 256, 14, 14,
        1, 20, 2000, 23);        
    model.list_layer[22]->setDstData(model.list_layer[23]->SrcData());

    model.list_layer[24] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 256, 14, 14,
        1, 1, 1, 1,
        256, 3, 3,
        MAX_BATCH_SIZE, 256, 14, 14,
        1, 20, 2000, 24);        
    model.list_layer[23]->setDstData(model.list_layer[24]->SrcData());

    model.list_layer[25] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 256, 14, 14,
        0, 0, 1, 1,
        1024, 1, 1,
        MAX_BATCH_SIZE, 1024, 14, 14,
        1, 20, 2000, 25);        
    model.list_layer[24]->setDstData(model.list_layer[25]->SrcData());

    for(int k=1; k<6; k++){
        model.list_layer[3*k+23] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 1024, 14, 14,
            0, 0, 1, 1,
            256, 1, 1,
            MAX_BATCH_SIZE, 256, 14, 14,
            1, 20, 2000, 3*k+23);        
        model.list_layer[3*k+22]->setDstData(model.list_layer[3*k+23]->SrcData());
    
        model.list_layer[3*k+24] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 256, 14, 14,
            1, 1, 1, 1,
            256, 3, 3,
            MAX_BATCH_SIZE, 256, 14, 14,
            1, 20, 2000, 3*k+24);        
        model.list_layer[3*k+23]->setDstData(model.list_layer[3*k+24]->SrcData());
    
        model.list_layer[3*k+25] = new Layer(CONV_RESIDUAL, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 256, 14, 14,
            0, 0, 1, 1,
            1024, 1, 1,
            MAX_BATCH_SIZE, 1024, 14, 14,
            1, 20, 2000, 3*k+25);        
        model.list_layer[3*k+24]->setDstData(model.list_layer[3*k+25]->SrcData());
    }

    /* fourth loop */
    model.list_layer[41] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 1024, 14, 14,
        0, 0, 2, 2,
        512, 1, 1,
        MAX_BATCH_SIZE, 512, 7, 7,
        1, 20, 2000, 41);        
    model.list_layer[40]->setDstData(model.list_layer[41]->SrcData());

    model.list_layer[42] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 512, 7, 7,
        1, 1, 1, 1,
        512, 3, 3,
        MAX_BATCH_SIZE, 512, 7, 7,
        1, 20, 2000, 42);        
    model.list_layer[41]->setDstData(model.list_layer[42]->SrcData());

    model.list_layer[43] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 512, 7, 7,
        0, 0, 1, 1,
        2048, 1, 1,
        MAX_BATCH_SIZE, 2048, 7, 7,
        1, 20, 2000, 43);        
    model.list_layer[42]->setDstData(model.list_layer[43]->SrcData());

    for(int k=1; k<3; k++){
        model.list_layer[3*k+41] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 2048, 7, 7,
            0, 0, 1, 1,
            512, 1, 1,
            MAX_BATCH_SIZE, 512, 7, 7,
            1, 20, 2000, 3*k+41);        
        model.list_layer[3*k+40]->setDstData(model.list_layer[3*k+41]->SrcData());
    
        model.list_layer[3*k+42] = new Layer(CONV, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 512, 7, 7,
            1, 1, 1, 1,
            512, 3, 3,
            MAX_BATCH_SIZE, 512, 7, 7,
            1, 20, 2000, 3*k+42);        
        model.list_layer[3*k+41]->setDstData(model.list_layer[3*k+42]->SrcData());
    
        model.list_layer[3*k+43] = new Layer(CONV_RESIDUAL, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 512, 7, 7,
            0, 0, 1, 1,
            2048, 1, 1,
            MAX_BATCH_SIZE, 2048, 7, 7,
            1, 20, 2000, 3*k+43);        
        model.list_layer[3*k+42]->setDstData(model.list_layer[3*k+43]->SrcData());
    }

    model.list_layer[50] = new Layer(POOL_AVERAGE, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem, 
        MAX_BATCH_SIZE, 2048, 7, 7,
        0, 0, 1, 1,
        2048, 7, 7,
        MAX_BATCH_SIZE, 2048, 1, 1,
        1, 20, 2000, 50);
    model.list_layer[49]->setDstData(model.list_layer[50]->SrcData());

    model.list_layer[51] = new Layer(CONV_LAST, &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 2048, 1, 1,
        0, 0, 1, 1,
        1, 1, 1,
        MAX_BATCH_SIZE, 2048, 1, 1,
        1, 20, 2000, 51);
    model.list_layer[50]->setDstData(model.list_layer[51]->SrcData());

    return model;
}