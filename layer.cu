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

#ifndef LAYER
#include "layer.h"
#define LAYER
#endif

#include "model.h"

__global__ void initGPUData_ker(float *data, int numElements, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
       data[tid] = value;
    }
}

void initGPUData(float *data, int numElements, float value) {
    dim3 gridDim;
    dim3 blockDim;
 
    blockDim.x = 1024;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
 
    initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}

Layer::Layer(
    enum layer_t         _layerType,
    cublasHandle_t* _cublasHandle,
    cudnnHandle_t*  _cudnnHandle,
    cudaStream_t*   _stream_compute,
    cudaStream_t*   _stream_memory,
    int _n, int _c, int _h, int _w, 
    int _pad_h, int _pad_w, int _stride_h, int _stride_w, 
    int _k, int _r, int _s,
    int _n_out, int _c_out,
    int _h_out, int _w_out,
    int _seqlength, int _hidden_size,
    int _num_layers,
    int _idx):  layerType(_layerType), 
                              cudnnHandle(_cudnnHandle), cublasHandle(_cublasHandle), stream_compute(_stream_compute), stream_memory(_stream_memory), 
                              n_in(_n), c_in(_c), h_in(_h), w_in(_w),
                              pad_h(_pad_h), pad_w(_pad_w), stride_h(_stride_h), stride_w(_stride_w),
                              k(_k), r(_r), s(_s),
                              n_out(_n_out), c_out(_c_out), h_out(_h_out), w_out(_w_out),
                              seqlength(_seqlength), hidden_size(_hidden_size), num_layers(_num_layers), idx(_idx)
  {
  
    // default    
    dataType      = DATA_PRECISION;
    modeConv      = CUDNN_CROSS_CORRELATION;
    tensorFormat  = CUDNN_TENSOR_NCHW;
    inPlaceOp     = false;
    
    srcData       = NULL;
    tempData      = NULL;
    filterData    = NULL;
    biasData      = NULL;
    dstData       = NULL;
    diffData      = NULL; // input to layer when backprop
    gradData      = NULL; // output of layer when backprop
    algo_int      = 2;
    fwdAlgo       = (cudnnConvolutionFwdAlgo_t) algo_int;
    softmaxAlgo   = CUDNN_SOFTMAX_FAST;
    softmaxMode   = CUDNN_SOFTMAX_MODE_INSTANCE;
  
    inputmodeRNN  = CUDNN_LINEAR_INPUT;
    modeRNN       = CUDNN_RNN_RELU;
    int bidirectional;
    if(idx == 0){
        direction     = CUDNN_BIDIRECTIONAL;
        bidirectional = 1;
    }
    else{
        direction = CUDNN_UNIDIRECTIONAL;
        bidirectional = 0;
    }
  
    rnnalgo_int   = 2;
    rnnAlgo       = CUDNN_RNN_ALGO_STANDARD;
  
    hx = NULL;
    cx = NULL;
    hy = NULL;
    cy = NULL;
  
    sizeInBytes = 0;
    workSpace = NULL;
  
    switch(_layerType) {
    case RNN_LAST:
      {
  
      }
    case RNN_DECODER:
      {
  
      }
    case RNN:
      {
          assert(n_in == n_out);
  
          cudaMalloc(&srcData, seqlength * hidden_size * n_in * sizeof(value_type));
          if(_layerType == RNN_LAST) cudaMalloc(&dstData, seqlength * hidden_size * n_out * sizeof(value_type));
          
          checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
          checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc));
          
          checkCUDNN(cudnnCreateTensorDescriptor(&hxDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&cxDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&hyDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&cyDesc));
  
          cudaMalloc(&tempData, seqlength * hidden_size * n_in * sizeof(value_type));
          cudaMalloc(&hx, num_layers * hidden_size * n_in * (bidirectional ? 2 : 1) * sizeof(value_type));
          cudaMalloc(&cx, num_layers * hidden_size * n_in * (bidirectional ? 2 : 1) * sizeof(value_type));
          cudaMalloc(&hy, num_layers * hidden_size * n_out * (bidirectional ? 2 : 1) * sizeof(value_type));
          cudaMalloc(&cy, num_layers * hidden_size * n_out * (bidirectional ? 2 : 1) * sizeof(value_type));
  
          xDesc = (cudnnTensorDescriptor_t* )malloc(seqlength * sizeof(cudnnTensorDescriptor_t));
          yDesc = (cudnnTensorDescriptor_t* )malloc(seqlength * sizeof(cudnnTensorDescriptor_t));
  
          checkCUDNN(cudnnCreateFilterDescriptor(&weightDesc));
          checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
  
          initGPUData((float*)srcData, seqlength * hidden_size * n_in, 1.f);
          if (hx != NULL) initGPUData((float*)hx, num_layers * hidden_size * n_in * (bidirectional ? 2 : 1), 1.f);
          if (cx != NULL) initGPUData((float*)cx, num_layers * hidden_size * n_in * (bidirectional ? 2 : 1), 1.f);
  
          change_size_RNN(1, n_in);
  
          checkCUDNN(cudnnGetRNNWorkspaceSize(*cudnnHandle, rnnDesc,
                                              seqlength, xDesc,
                                              &sizeInBytes));
          // std::cout<<"[Note] RNN WorkingSpace required: "<<sizeInBytes<<" (bytes)"<<std::endl;
          if(sizeInBytes!=0)   cudaMalloc(&workSpace, sizeInBytes);
  
      }
      break;
    case ATTENTION:
      {
          checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));   
          checkCUDNN(cudnnCreateTensorDescriptor(&gemmTensorDesc));
          checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&weightTensorDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&softmaxTensorDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&contextTensorDesc));
  
          checkCuda(cudaMalloc(&srcData, n_in * seqlength * hidden_size * sizeof(value_type)));
          checkCuda(cudaMalloc(&dstData, n_in * seqlength * hidden_size * sizeof(value_type)));
          checkCuda(cudaMalloc(&gemmData, n_in * seqlength * hidden_size * sizeof(value_type)));
          checkCuda(cudaMalloc(&weightData, n_in* 1 * seqlength * sizeof(value_type)));
          checkCuda(cudaMalloc(&softmaxData, n_in* 1 * seqlength * sizeof(value_type)));
          checkCuda(cudaMalloc(&contextData, n_in* 1 * hidden_size * sizeof(value_type)));
          
          
          // checkCUDNN(cudnnGetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));        
  
          checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));
          checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in/1, 1, hidden_size, seqlength));
          checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_in/1, 1, hidden_size, seqlength));
          checkCUDNN(cudnnSetTensor4dDescriptor(gemmTensorDesc, tensorFormat, dataType, n_in/1, 1, hidden_size, seqlength));
          checkCUDNN(cudnnSetTensor4dDescriptor(weightTensorDesc, tensorFormat, dataType, n_in/1, 1, 1, seqlength));
          checkCUDNN(cudnnSetTensor4dDescriptor(softmaxTensorDesc, tensorFormat, dataType, n_in/1, 1, 1, seqlength));
          checkCUDNN(cudnnSetTensor4dDescriptor(contextTensorDesc, tensorFormat, dataType, n_in/1, 1, 1, hidden_size));
      }
      break;
    case CONV:
      {
        // create cudnn descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
  
        value_type *filterData_h	= (value_type*)malloc(k*c_in*r*s*sizeof(value_type));
  
        for(unsigned i=0; i<k*c_in*r*s; i++)	filterData_h[i]	= (value_type)i/(k*c_in*r*s);
        checkCuda(cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type)));
      //   cudaMalloc(&dstData, n_out*c_out*h_out*w_out*sizeof(value_type));
        checkCuda(cudaMalloc(&filterData, k*c_in*r*s*sizeof(value_type)));
        checkCuda(cudaMalloc(&tempData, n_in*c_in*h_in*w_in*sizeof(value_type)));
  
        checkCuda(cudaMemcpy(filterData, filterData_h, k*c_in*r*s*sizeof(value_type), cudaMemcpyHostToDevice));
        // free memory
        free(filterData_h);
  
        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in/1, c_in, h_in, w_in));
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat, k, c_in, r, s));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_w, stride_h, 1, 1, modeConv, CUDNN_DATA_FLOAT));
        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, tensorFormat, dataType, n_in/1, c_in, h_in, w_in));
        // find dimension of convolution output
  
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));
        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
  
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*cudnnHandle,
                                  srcTensorDesc,
                                  filterDesc,
                                  convDesc,
                                  dstTensorDesc,
                                  fwdAlgo,
                                  &sizeInBytes));
      //   std::cout<<"[Note] CONV WorkingSpace required: "<<sizeInBytes<<" (bytes)"<<std::endl;
        if(sizeInBytes!=0)  checkCuda(cudaMalloc(&workSpace, sizeInBytes));
  
  
      }
      break;
    case CONV_RESIDUAL:
      {
        // create cudnn descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
  
        value_type *filterData_h	= (value_type*)malloc(k*c_in*r*s*sizeof(value_type));
  
        for(unsigned i=0; i<k*c_in*r*s; i++)	filterData_h[i]	= (value_type)i/(k*c_in*r*s);
        checkCuda(cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type)));
      //   checkCuda(cudaMalloc(&biasData, n_in*c_in*h_in*w_in*sizeof(value_type)));
      //   cudaMalloc(&dstData, n_out*c_out*h_out*w_out*sizeof(value_type));
        checkCuda(cudaMalloc(&filterData, k*c_in*r*s*sizeof(value_type)));
        checkCuda(cudaMalloc(&tempData, n_in*c_in*h_in*w_in*sizeof(value_type)));
  
        checkCuda(cudaMemcpy(filterData, filterData_h, k*c_in*r*s*sizeof(value_type), cudaMemcpyHostToDevice));
        // free memory
        free(filterData_h);
  
        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in/1, c_in, h_in, w_in));
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat, k, c_in, r, s));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_w, stride_h, 1, 1, modeConv, CUDNN_DATA_FLOAT));
        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, tensorFormat, dataType, n_out/1, c_out, h_out, w_out));
        // find dimension of convolution output
  
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));
        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
  
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*cudnnHandle,
          srcTensorDesc,
          filterDesc,
          convDesc,
          dstTensorDesc,
          fwdAlgo,
          &sizeInBytes));
      // std::cout<<"[Note] CONV WorkingSpace required: "<<sizeInBytes<<" (bytes)"<<std::endl;
      if(sizeInBytes!=0)  checkCuda(cudaMalloc(&workSpace, sizeInBytes));
  
      }
      break;
  
    case CONV_LAST:
      {
        // create cudnn descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
  
        value_type *filterData_h	= (value_type*)malloc(k*c_in*r*s*sizeof(value_type));
  
        for(unsigned i=0; i<k*c_in*r*s; i++)	filterData_h[i]	= (value_type)i/(k*c_in*r*s);
        cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type));
  
        cudaMalloc(&filterData, k*c_in*r*s*sizeof(value_type));
        cudaMalloc(&tempData, n_in*c_in*h_in*w_in*sizeof(value_type));
  
        checkCuda(cudaMemcpy(filterData, filterData_h, k*c_in*r*s*sizeof(value_type), cudaMemcpyHostToDevice));
        // free memory
        free(filterData_h);
  
        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in/1, c_in, h_in, w_in));
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat, k, c_in, r, s));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_w, stride_h, 1, 1, modeConv, CUDNN_DATA_FLOAT));
        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, tensorFormat, dataType, n_out/1, c_out, h_out, w_out));
        // find dimension of convolution output
  
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));
  
        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
        cudaMalloc(&dstData, n_out*c_out*h_out*w_out*sizeof(value_type));
  
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*cudnnHandle,
          srcTensorDesc,
          filterDesc,
          convDesc,
          dstTensorDesc,
          fwdAlgo,
          &sizeInBytes));
      // std::cout<<"[Note] CONV WorkingSpace required: "<<sizeInBytes<<" (bytes)"<<std::endl;
      if(sizeInBytes!=0)  checkCuda(cudaMalloc(&workSpace, sizeInBytes));
  
      }
      break;
  
    case RELU:
        // create cudnn descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
  
        assert(n_in == n_out);
        assert(c_in == c_out);
        assert(h_in == h_out);
        assert(w_in == w_out);
  
        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));
  
        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
        // default behavior of RELU is to do in-place op
        this->inPlaceOp = true;
      break;
    case POOL:
          checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
          checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
          checkCuda(cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type)));
  
          // set input descriptors
          checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in/1, c_in, h_in, w_in));
          checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, r, s, pad_h, pad_w, stride_h, stride_w));
          // find dimension of pooling output
          checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolingDesc, srcTensorDesc, &n_out, &c_out, &h_out, &w_out));
  
          // set output descriptor based on above
          checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
      break;
    case POOL_AVERAGE:
          checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
          checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
          checkCuda(cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type)));
  
          // set input descriptors
          checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in/1, c_in, h_in, w_in));
          checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, r, s, pad_h, pad_w, stride_h, stride_w));
          // find dimension of pooling output
          checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolingDesc, srcTensorDesc, &n_out, &c_out, &h_out, &w_out));
  
          // set output descriptor based on above
          checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
        break;
    case SOFTMAX:
          checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));   
  
          checkCuda(cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type)));
  
          checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));
          checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
          
      break;
    case ACTIVATE:
          checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));   
          checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
  
          checkCuda(cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type)));
          
          // checkCUDNN(cudnnGetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));        
  
          checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));
          checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));
          checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
  
      break;        
  
    default:
      printf("WARNING : layer %d\n", _layerType);
      assert(0);
      break;
    }
  
    // create descriptor for backprop
    checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}

void Layer::change_size_RNN(int initialized, int size){
    int bidirectional;
    if(idx == 0){
        direction     = CUDNN_BIDIRECTIONAL;
        bidirectional = 1;
    }
    else{
        direction = CUDNN_UNIDIRECTIONAL;
        bidirectional = 0;
    }

    int dimA[3];
    int strideA[3];

    for(int i=0; i<seqlength; i++){
        checkCUDNN(cudnnCreateTensorDescriptor(&xDesc[i]));
        checkCUDNN(cudnnCreateTensorDescriptor(&yDesc[i]));

        dimA[0] = size;
        dimA[1] = hidden_size;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        cudnnSetTensorNdDescriptor(xDesc[i], dataType, 3, dimA, strideA);

        dimA[0] = size;
        dimA[1] = bidirectional ? 2*hidden_size : hidden_size;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        cudnnSetTensorNdDescriptor(yDesc[i], dataType, 3, dimA, strideA);
    }

    dimA[0] = num_layers * (bidirectional ? 2 : 1);
    dimA[1] = size;
    dimA[2] = hidden_size;
    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    checkCUDNN(cudnnSetTensorNdDescriptor(hxDesc, dataType, 3, dimA, strideA));
    checkCUDNN(cudnnSetTensorNdDescriptor(cxDesc, dataType, 3, dimA, strideA));
    checkCUDNN(cudnnSetTensorNdDescriptor(hyDesc, dataType, 3, dimA, strideA));
    checkCUDNN(cudnnSetTensorNdDescriptor(cyDesc, dataType, 3, dimA, strideA));

    unsigned long long seed = 1337ull;

    checkCUDNN(cudnnDropoutGetStatesSize(*cudnnHandle, &stateSize));

    // cudaMalloc(&states, stateSize);
    states = NULL;
    stateSize = 0;

    checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc,
                                    *cudnnHandle,
                                    0,
                                    states,
                                    stateSize,
                                    seed));
    

    checkCUDNN(cudnnSetRNNDescriptor(*cudnnHandle, rnnDesc, hidden_size, num_layers, dropoutDesc, inputmodeRNN, direction, modeRNN, rnnAlgo, dataType));

    checkCUDNN(cudnnGetRNNParamsSize(*cudnnHandle, rnnDesc, xDesc[0], &weightsSize, dataType));

    int dimW[3];
    dimW[0] = weightsSize / sizeof(float);
    dimW[1] = 1;
    dimW[2] = 1;
    checkCUDNN(cudnnSetFilterNdDescriptor(weightDesc, dataType, tensorFormat, 3, dimW));
    if(initialized) cudaMalloc(&weight, weightsSize);

    int numLinearLayers = 2;
    for(int layer=0; layer<num_layers * (bidirectional ? 2 : 1); layer++){
        for(int linLayerID=0; linLayerID < numLinearLayers; linLayerID++){
            cudnnFilterDescriptor_t linLayerMatDesc;
            checkCUDNN(cudnnCreateFilterDescriptor(&linLayerMatDesc));
            float* linLayerMat;

            checkCUDNN(cudnnGetRNNLinLayerMatrixParams(*cudnnHandle, rnnDesc, layer, xDesc[0], weightDesc, weight, linLayerID, linLayerMatDesc, (void**)&linLayerMat));
            int nbDims;
            int filterDimA[3];
            checkCUDNN(cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType, &tensorFormat, &nbDims, filterDimA));
            checkCUDNN(cudnnDestroyFilterDescriptor(linLayerMatDesc));
            cudnnFilterDescriptor_t linLayerBiasDesc;
            checkCUDNN(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
            float* linLayerBias;
            
            checkCUDNN(cudnnGetRNNLinLayerBiasParams(*cudnnHandle, rnnDesc, layer, xDesc[0], weightDesc, weight, linLayerID, linLayerBiasDesc, (void**)&linLayerBias));
            checkCUDNN(cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, &dataType, &tensorFormat, &nbDims, filterDimA));
            checkCUDNN(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
        }
    }

    return;
}