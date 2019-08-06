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

#include "layer.h"
#include "model.h"

using namespace std;

enum layer_t model_info[] = {CONV_LAST};

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

#define ASSERT_EQ(A, B) {  \
  if((A)!=(B)) { printf("\n\n[CNMEM FAILED]\n"); this->printCnmemMemoryUsage(); assert(0); }        \
}

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCUBLAS(status) {                                          \
    std::stringstream _error;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                              \
      _error << "CUBLAS failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      assert(0);                                                        \
      FatalError(_error.str());                                        \
    }                                                                  \
}

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
  }
  return result;
}

Layer::Layer(
  enum layer_t         _layerType,
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
                            cudnnHandle(_cudnnHandle), stream_compute(_stream_compute), stream_memory(_stream_memory), 
                            n_in(_n), c_in(_c), h_in(_h), w_in(_w),
                            pad_h(_pad_h), pad_w(_pad_w), stride_h(_stride_h), stride_w(_stride_w),
                            k(_k), r(_r), s(_s),
                            n_out(_n_out), c_out(_c_out), h_out(_h_out), w_out(_w_out),
                            seqlength(_seqlength), hidden_size(_hidden_size), num_layers(_num_layers), idx(_idx)
{

  // default    
  dataType      = DATA_PRECISION;\
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

        int dimA[3];
        int strideA[3];

        for(int i=0; i<seqlength; i++){
            checkCUDNN(cudnnCreateTensorDescriptor(&xDesc[i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&yDesc[i]));

            dimA[0] = n_in;
            dimA[1] = hidden_size;
            dimA[2] = 1;
            strideA[0] = dimA[2] * dimA[1];
            strideA[1] = dimA[2];
            strideA[2] = 1;

            cudnnSetTensorNdDescriptor(xDesc[i], dataType, 3, dimA, strideA);

            dimA[0] = n_in;
            dimA[1] = bidirectional ? 2*hidden_size : hidden_size;
            dimA[2] = 1;

            strideA[0] = dimA[2] * dimA[1];
            strideA[1] = dimA[2];
            strideA[2] = 1;

            cudnnSetTensorNdDescriptor(yDesc[i], dataType, 3, dimA, strideA);
        }

        dimA[0] = num_layers * (bidirectional ? 2 : 1);
        dimA[1] = n_in;
        dimA[2] = hidden_size;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        checkCUDNN(cudnnSetTensorNdDescriptor(hxDesc, dataType, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(cxDesc, dataType, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(hyDesc, dataType, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(cyDesc, dataType, 3, dimA, strideA));

        checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));

        unsigned long long seed = 1337ull;

        checkCUDNN(cudnnDropoutGetStatesSize(*cudnnHandle, &stateSize));
        cudaMalloc(&states, stateSize);
        checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc,
                                        *cudnnHandle,
                                        0,
                                        states,
                                        stateSize,
                                        seed));

        checkCUDNN(cudnnSetRNNDescriptor(*cudnnHandle, rnnDesc, hidden_size, num_layers, dropoutDesc, inputmodeRNN, direction, modeRNN, rnnAlgo, dataType));

        checkCUDNN(cudnnCreateFilterDescriptor(&weightDesc));

        checkCUDNN(cudnnGetRNNParamsSize(*cudnnHandle, rnnDesc, xDesc[0], &weightsSize, dataType));

        int dimW[3];
        dimW[0] = weightsSize / sizeof(float);
        dimW[1] = 1;
        dimW[2] = 1;
        checkCUDNN(cudnnSetFilterNdDescriptor(weightDesc, dataType, tensorFormat, 3, dimW));
        cudaMalloc(&weight, weightsSize);

        initGPUData((float*)srcData, seqlength * hidden_size * n_in, 1.f);
        if (hx != NULL) initGPUData((float*)hx, num_layers * hidden_size * n_in * (bidirectional ? 2 : 1), 1.f);
        if (cx != NULL) initGPUData((float*)cx, num_layers * hidden_size * n_in * (bidirectional ? 2 : 1), 1.f);
     
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

        checkCUDNN(cudnnGetRNNWorkspaceSize(*cudnnHandle, rnnDesc,
                                            seqlength, xDesc,
                                            &sizeInBytes));
        // std::cout<<"[Note] RNN WorkingSpace required: "<<sizeInBytes<<" (bytes)"<<std::endl;
        if(sizeInBytes!=0)   cudaMalloc(&workSpace, sizeInBytes);


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

  default:
    printf("layer %d\n", _layerType);
    assert(0);
    break;
  }

  // create descriptor for backprop
  checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}

struct Model_layer create_GNMT(cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem){
    struct Model_layer model;

    int seq_length_GNMT = 50;
    int hidden_size_GNMT = 1024;
    int num_layers_GNMT = 1;

    model.list_layer[0] = new Layer(RNN, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        seq_length_GNMT, hidden_size_GNMT, num_layers_GNMT, 0);

    model.list_layer[1] = new Layer(RNN, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        hidden_size_GNMT*2, hidden_size_GNMT, num_layers_GNMT, 1);
    model.list_layer[0]->setDstData(model.list_layer[1]->SrcData());

    model.list_layer[2] = new Layer(RNN, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        hidden_size_GNMT, hidden_size_GNMT, num_layers_GNMT, 2);
    model.list_layer[1]->setDstData(model.list_layer[2]->SrcData());

    model.list_layer[3] = new Layer(RNN_LAST, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        MAX_BATCH_SIZE, 0, 0, 0,
        hidden_size_GNMT, hidden_size_GNMT, num_layers_GNMT, 3);
    model.list_layer[2]->setDstData(model.list_layer[3]->SrcData());


    return model;
}

struct Model_layer create_Resnet(cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem){
    struct Model_layer model;
    model.list_layer[0] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 3, 224, 224,
        3, 3, 2, 2,
        64, 7, 7,
        MAX_BATCH_SIZE, 64, 112, 112,
        1, 20, 2000, 0);

    model.list_layer[1] = new Layer(POOL, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 64, 112, 112,
        0, 0, 2, 2,
        64, 3, 3,
        MAX_BATCH_SIZE, 64, 56, 56,
        1, 20, 2000, 1);
    model.list_layer[0]->setDstData(model.list_layer[1]->SrcData());

    /* loop */
    
    model.list_layer[2] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 64, 56, 56,
        0, 0, 1, 1,
        64, 1, 1,
        MAX_BATCH_SIZE, 64, 56, 56,
        1, 20, 2000, 2);

    model.list_layer[1]->setDstData(model.list_layer[2]->SrcData());

    model.list_layer[3] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 64, 56, 56,
        1, 1, 1, 1,
        64, 3, 3,
        MAX_BATCH_SIZE, 64, 56, 56,
        1, 20, 2000, 3);

    model.list_layer[2]->setDstData(model.list_layer[3]->SrcData());

    model.list_layer[4] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 64, 56, 56,
        0, 0, 1, 1,
        256, 1, 1,
        MAX_BATCH_SIZE, 256, 56, 56,
        1, 20, 2000, 4);

    model.list_layer[3]->setDstData(model.list_layer[4]->SrcData());

    for(int k=1; k<3; k++){

        model.list_layer[3*k+2] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 256, 56, 56,
            0, 0, 1, 1,
            64, 1, 1,
            MAX_BATCH_SIZE, 64, 56, 56,
            1, 20, 2000, 3*k+2);

        model.list_layer[3*k+1]->setDstData(model.list_layer[3*k+2]->SrcData());

        model.list_layer[3*k+3] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 64, 56, 56,
            1, 1, 1, 1,
            64, 3, 3,
            MAX_BATCH_SIZE, 64, 56, 56,
            1, 20, 2000, 3*k+3);

        model.list_layer[3*k+2]->setDstData(model.list_layer[3*k+3]->SrcData());

        model.list_layer[3*k+4] = new Layer(CONV_RESIDUAL, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 64, 56, 56,
            0, 0, 1, 1,
            256, 1, 1,
            MAX_BATCH_SIZE, 256, 56, 56,
            1, 20, 2000, 3*k+4);

        model.list_layer[3*k+3]->setDstData(model.list_layer[3*k+4]->SrcData());
    }

    /* second loop 1 */

    model.list_layer[11] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 256, 56, 56,
        0, 0, 2, 2,
        128, 1, 1,
        MAX_BATCH_SIZE, 128, 28, 28,
        1, 20, 2000, 11);
    model.list_layer[10]->setDstData(model.list_layer[11]->SrcData());

    model.list_layer[12] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 128, 28, 28,
        1, 1, 1, 1,
        128, 3, 3,
        MAX_BATCH_SIZE, 128, 28, 28,
        1, 20, 2000, 12);
    model.list_layer[11]->setDstData(model.list_layer[12]->SrcData());

    model.list_layer[13] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 128, 28, 28,
        0, 0, 1, 1,
        512, 1, 1,
        MAX_BATCH_SIZE, 512, 28, 28,
        1, 20, 2000, 13);        
    model.list_layer[12]->setDstData(model.list_layer[13]->SrcData());

    for(int k=1; k<4; k++){
        model.list_layer[3*k+11] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 512, 28, 28,
            0, 0, 1, 1,
            128, 1, 1,
            MAX_BATCH_SIZE, 128, 28, 28,
            1, 20, 2000, 3*k+11);
        model.list_layer[3*k+10]->setDstData(model.list_layer[3*k+11]->SrcData());

        model.list_layer[3*k+12] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 128, 28, 28,
            1, 1, 1, 1,
            128, 3, 3,
            MAX_BATCH_SIZE, 128, 28, 28,
            1, 20, 2000, 3*k+12);
        model.list_layer[3*k+11]->setDstData(model.list_layer[3*k+12]->SrcData());

        model.list_layer[3*k+13] = new Layer(CONV_RESIDUAL, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 128, 28, 28,
            0, 0, 1, 1,
            512, 1, 1,
            MAX_BATCH_SIZE, 512, 28, 28,
            1, 20, 2000, 3*k+13);        
        model.list_layer[3*k+12]->setDstData(model.list_layer[3*k+13]->SrcData());
    }

    /* third loop */
    model.list_layer[23] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 512, 28, 28,
        0, 0, 2, 2,
        256, 1, 1,
        MAX_BATCH_SIZE, 256, 14, 14,
        1, 20, 2000, 23);        
    model.list_layer[22]->setDstData(model.list_layer[23]->SrcData());

    model.list_layer[24] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 256, 14, 14,
        1, 1, 1, 1,
        256, 3, 3,
        MAX_BATCH_SIZE, 256, 14, 14,
        1, 20, 2000, 24);        
    model.list_layer[23]->setDstData(model.list_layer[24]->SrcData());

    model.list_layer[25] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 256, 14, 14,
        0, 0, 1, 1,
        1024, 1, 1,
        MAX_BATCH_SIZE, 1024, 14, 14,
        1, 20, 2000, 25);        
    model.list_layer[24]->setDstData(model.list_layer[25]->SrcData());

    for(int k=1; k<6; k++){
        model.list_layer[3*k+23] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 1024, 14, 14,
            0, 0, 1, 1,
            256, 1, 1,
            MAX_BATCH_SIZE, 256, 14, 14,
            1, 20, 2000, 3*k+23);        
        model.list_layer[3*k+22]->setDstData(model.list_layer[3*k+23]->SrcData());
    
        model.list_layer[3*k+24] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 256, 14, 14,
            1, 1, 1, 1,
            256, 3, 3,
            MAX_BATCH_SIZE, 256, 14, 14,
            1, 20, 2000, 3*k+24);        
        model.list_layer[3*k+23]->setDstData(model.list_layer[3*k+24]->SrcData());
    
        model.list_layer[3*k+25] = new Layer(CONV_RESIDUAL, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 256, 14, 14,
            0, 0, 1, 1,
            1024, 1, 1,
            MAX_BATCH_SIZE, 1024, 14, 14,
            1, 20, 2000, 3*k+25);        
        model.list_layer[3*k+24]->setDstData(model.list_layer[3*k+25]->SrcData());
    }

    /* fourth loop */
    model.list_layer[41] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 1024, 14, 14,
        0, 0, 2, 2,
        512, 1, 1,
        MAX_BATCH_SIZE, 512, 7, 7,
        1, 20, 2000, 41);        
    model.list_layer[40]->setDstData(model.list_layer[41]->SrcData());

    model.list_layer[42] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 512, 7, 7,
        1, 1, 1, 1,
        512, 3, 3,
        MAX_BATCH_SIZE, 512, 7, 7,
        1, 20, 2000, 42);        
    model.list_layer[41]->setDstData(model.list_layer[42]->SrcData());

    model.list_layer[43] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 512, 7, 7,
        0, 0, 1, 1,
        2048, 1, 1,
        MAX_BATCH_SIZE, 2048, 7, 7,
        1, 20, 2000, 43);        
    model.list_layer[42]->setDstData(model.list_layer[43]->SrcData());

    for(int k=1; k<3; k++){
        model.list_layer[3*k+41] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 2048, 7, 7,
            0, 0, 1, 1,
            512, 1, 1,
            MAX_BATCH_SIZE, 512, 7, 7,
            1, 20, 2000, 3*k+41);        
        model.list_layer[3*k+40]->setDstData(model.list_layer[3*k+41]->SrcData());
    
        model.list_layer[3*k+42] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 512, 7, 7,
            1, 1, 1, 1,
            512, 3, 3,
            MAX_BATCH_SIZE, 512, 7, 7,
            1, 20, 2000, 3*k+42);        
        model.list_layer[3*k+41]->setDstData(model.list_layer[3*k+42]->SrcData());
    
        model.list_layer[3*k+43] = new Layer(CONV_RESIDUAL, &cudnnHandle, &myStream_compute, &myStream_mem,
            MAX_BATCH_SIZE, 512, 7, 7,
            0, 0, 1, 1,
            2048, 1, 1,
            MAX_BATCH_SIZE, 2048, 7, 7,
            1, 20, 2000, 3*k+43);        
        model.list_layer[3*k+42]->setDstData(model.list_layer[3*k+43]->SrcData());
    }

    model.list_layer[50] = new Layer(POOL_AVERAGE, &cudnnHandle, &myStream_compute, &myStream_mem, 
        MAX_BATCH_SIZE, 2048, 7, 7,
        0, 0, 1, 1,
        2048, 7, 7,
        MAX_BATCH_SIZE, 2048, 1, 1,
        1, 20, 2000, 50);
    model.list_layer[49]->setDstData(model.list_layer[50]->SrcData());

    model.list_layer[51] = new Layer(CONV_LAST, &cudnnHandle, &myStream_compute, &myStream_mem,
        MAX_BATCH_SIZE, 2048, 1, 1,
        0, 0, 1, 1,
        1, 1, 1,
        MAX_BATCH_SIZE, 2048, 1, 1,
        1, 20, 2000, 51);
    model.list_layer[50]->setDstData(model.list_layer[51]->SrcData());

    return model;
}

struct Model_layer createModel(int type, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem){
    struct Model_layer model;
    printf("type : %d\n", type);
    
    if(type == 1){
        model = create_Resnet(cudnnHandle, myStream_compute, myStream_mem);
    }
    else if(type == 2){
        model = create_GNMT(cudnnHandle, myStream_compute, myStream_mem);
    }
    else{
        
        for(int i=0; i<MAX_LAYER_NUM; i++){
            model.list_layer[i] = new Layer(model_info[i], &cudnnHandle, &myStream_compute, &myStream_mem,
                MAX_BATCH_SIZE, 64, 224, 224, 1, 1, 1, 1, 64, 3, 3,
                MAX_BATCH_SIZE, 64, 112, 112, 1, 20, 2000, i);
            if(i>0) model.list_layer[i-1]->setDstData(model.list_layer[i]->SrcData());
        }
    }
    model.data_first = 0;
    model.data_num = 0;
    model.data_exist = 0;

    return model;
}


bool make_delay(void){
    return true;
}

double get_current_time(clock_t start){
    clock_t now = clock();
    return (double)(now-start)/ CLOCKS_PER_SEC*1000;
}

struct vector_layer{
    int idx_layer;
    vector<int> idx_request;
    int seqlength;
};

struct vector_layer set_vector(int idx_layer, int input_index, int seqlength){
    struct vector_layer new_vector;
    new_vector.idx_layer = idx_layer;
    new_vector.idx_request.push_back(input_index);
    new_vector.seqlength = seqlength;
    return new_vector;
}

bool merge_request(vector<struct vector_layer>* v_layer){
    int idx_end = (*v_layer).size()-1;
    if(idx_end <= 0) return false;
    struct vector_layer v_1 = (*v_layer)[idx_end];
    struct vector_layer v_2 = (*v_layer)[idx_end-1];
    if(v_1.idx_layer == v_2.idx_layer){
        printf("before merge, request size : %d, current size - %d\n", v_2.idx_request.size(), (*v_layer).size());
        v_2.idx_request.insert(v_2.idx_request.end(), v_1.idx_request.begin(), v_1.idx_request.end());
        (*v_layer).pop_back();
        (*v_layer).pop_back();
        (*v_layer).push_back(v_2);

        printf("after merge, request size : %d, current size - %d\n", v_2.idx_request.size(), (*v_layer).size());
        return true;
    }
    else return false;


}



int main(int argc, char **argv)
{

    ifstream in(argv[1]);
    string s;
    char buf[100];

    vector<float> request_vector;
    vector<int> request_unroll;
    vector<int> request_current_unroll;
    int request_number;

    /* for data */
    vector<double> request_latency;

    if(in.is_open()){
        while(in){
            in.getline(buf, 100, ',');
            if(strlen(buf)) request_vector.push_back(atof(buf));
        }

        request_number = request_vector.size();
        printf("request num : %d\n", request_number);

        for(int i=0; i<request_number; i++){
            request_unroll.push_back(4);
            request_current_unroll.push_back(0);
            request_latency.push_back(-request_vector.at(i));
        }

    }
    else exit(-1);

    /* ---------------------------- setup ---------------------------- */
    clock_t start;

    int input_index = 0;

    // create streams
    cudaStream_t myStream_compute;
    cudaStreamCreate(&myStream_compute);

    cudaStream_t myStream_mem;
    cudaStreamCreate(&myStream_mem);
	// cudnn handle
    cudnnHandle_t         cudnnHandle;  
    checkCUDNN(cudnnCreate(&cudnnHandle));

    // link cudnnHandle with this stream
    checkCUDNN(cudnnSetStream(cudnnHandle, myStream_compute));
    checkCUDNN(cudnnSetStream(cudnnHandle, myStream_mem));

    vector<struct vector_layer> v_layer;
    struct Model_layer model = createModel(MODEL_TYPE, cudnnHandle, myStream_compute, myStream_mem);
    value_type alpha = value_type(1);
    value_type beta  = value_type(0);
    
    start = clock();

    /* --------------------------------------------------------------- */
    /* --------------------------------------------------------------- */
    /* --------------------------------------------------------------- */
    bool stall = false;
    while(input_index <= request_number){
        
        double input_time = get_current_time(start);

        stall = false;
        if(model.data_num >= MAX_REQ_SIZE)    stall = true;
        
        /* New input condition */
        if(stall==false && input_index != request_number && input_time > request_vector.at(input_index)){ 
            
            struct vector_layer v_new = set_vector(0, input_index, request_unroll.at(input_index) );
            
            v_layer.push_back(v_new);

            input_index += 1;
            printf("--came at time : %lf--\n", input_time);
            printf("--input index : %d\n", input_index);
        
            model.data_num += 1;

            if(merge_request(&v_layer)){
                printf("merged!\n");
            }
        }
        else{

            if(v_layer.size() <= 0) continue;

            std::cout<<"current req num: "<<model.data_num<<", starts at "<<model.data_first<<std::endl;

            int idx_end = v_layer.size()-1;
            struct vector_layer v_now = v_layer[idx_end];
            v_layer.pop_back();

            Layer* current_index_layer = model.list_layer[v_now.idx_layer];
            if(current_index_layer->layerType == CONV || current_index_layer->layerType == CONV_LAST || current_index_layer->layerType == CONV_RESIDUAL){
                int num_req = v_now.idx_request.size();
                printf("num req : %d\n", num_req);
                checkCUDNN(cudnnSetTensor4dDescriptor(current_index_layer->srcTensorDesc, current_index_layer->tensorFormat, current_index_layer->dataType, current_index_layer->n_in/MAX_BATCH_SIZE*num_req, current_index_layer->c_in, current_index_layer->h_in, current_index_layer->w_in));
                checkCUDNN(cudnnGetConvolution2dForwardOutputDim(current_index_layer->convDesc, current_index_layer->srcTensorDesc, current_index_layer->filterDesc, &current_index_layer->n_out, &current_index_layer->c_out, &current_index_layer->h_out, &current_index_layer->w_out));
                checkCUDNN(cudnnSetTensor4dDescriptor(current_index_layer->biasTensorDesc, current_index_layer->tensorFormat, current_index_layer->dataType, current_index_layer->n_out, current_index_layer->c_out, current_index_layer->h_out, current_index_layer->w_out));
                checkCUDNN(cudnnSetTensor4dDescriptor(current_index_layer->dstTensorDesc, current_index_layer->tensorFormat, current_index_layer->dataType, current_index_layer->n_out, current_index_layer->c_out, current_index_layer->h_out, current_index_layer->w_out));


                checkCUDNN(cudnnConvolutionForward( cudnnHandle, &alpha, current_index_layer->srcTensorDesc,
                                                current_index_layer->srcData,
                                                current_index_layer->filterDesc,
                                                current_index_layer->filterData,
                                                current_index_layer->convDesc,
                                                current_index_layer->fwdAlgo,
                                                current_index_layer->workSpace,
                                                current_index_layer->sizeInBytes,
                                                &beta,
                                                current_index_layer->dstTensorDesc,
                                                current_index_layer->dstData));

                if(current_index_layer->layerType == CONV_RESIDUAL){
                    Layer* bias_layer = model.list_layer[current_index_layer->idx -3];
                    current_index_layer->biasData = bias_layer->dstData;
                    
                    checkCUDNN(cudnnAddTensor( cudnnHandle, 
                        &alpha, 
                        current_index_layer->biasTensorDesc,
                        current_index_layer->biasData,
                        &beta,
                        current_index_layer->dstTensorDesc,
                        current_index_layer->dstData
                      ));
                }
                double now = get_current_time(start);
                printf("Passed index %d at time %lfms\n\n", v_now.idx_layer, now);
                v_now.idx_layer += 1;

                if(v_now.idx_layer >= MAX_LAYER_NUM){
                    int request_done = v_now.idx_request.size();
                    model.data_first += request_done;
                    model.data_num -= request_done;

                    vector<int>::iterator iter=v_now.idx_request.begin();
                
                    for(; iter!=v_now.idx_request.end(); iter++){

                        request_latency.at(*iter) += now;
                        printf("%lfms, latency of request no.%d : %lfms\n\n", now, *iter, request_latency.at(*iter));
                    }

                    if(input_index == request_number) break;
                }
                else{
                    v_layer.push_back(v_now);
                    if(merge_request(&v_layer)){
                        printf("merged!\n");
                    }
                }
            }
            else if(current_index_layer->layerType == POOL || current_index_layer->layerType == POOL_AVERAGE){

                std::cout<<"[Note] Pooling Layer"<<std::endl;
                
                int num_req = v_now.idx_request.size();
                printf("num req : %d\n", num_req);
                checkCUDNN(cudnnSetTensor4dDescriptor(current_index_layer->srcTensorDesc, current_index_layer->tensorFormat, current_index_layer->dataType, current_index_layer->n_in/MAX_BATCH_SIZE*num_req, current_index_layer->c_in, current_index_layer->h_in, current_index_layer->w_in));
                checkCUDNN(cudnnGetPooling2dForwardOutputDim(current_index_layer->poolingDesc, current_index_layer->srcTensorDesc, &current_index_layer->n_out, &current_index_layer->c_out, &current_index_layer->h_out, &current_index_layer->w_out));
                checkCUDNN(cudnnSetTensor4dDescriptor(current_index_layer->dstTensorDesc, current_index_layer->tensorFormat, current_index_layer->dataType, current_index_layer->n_out, current_index_layer->c_out, current_index_layer->h_out, current_index_layer->w_out));

                checkCUDNN(cudnnPoolingForward( cudnnHandle, current_index_layer->poolingDesc, &alpha, current_index_layer->srcTensorDesc,
                    current_index_layer->srcData,
                    &beta,
                    current_index_layer->dstTensorDesc,
                    current_index_layer->dstData));
                double now = get_current_time(start);
                printf("Passed index %d at time %lfms\n\n", v_now.idx_layer, now);
                v_now.idx_layer += 1;
                
                if(v_now.idx_layer >= MAX_LAYER_NUM){
                    int request_done = v_now.idx_request.size();
                    model.data_first += request_done;
                    model.data_num -= request_done;

                    if(input_index == request_number) break;
                }
                else{
                    v_layer.push_back(v_now);
                    if(merge_request(&v_layer)){
                        printf("merged!\n");
                    }
                }
            }
            else{

                if(idx_end>0){
                    struct vector_layer v_next = v_layer[idx_end-1];
                    if(v_next.idx_layer == v_now.idx_layer +1){
                        // int n_in = model.list_layer[v_next.idx_layer]->n_in;
                        // int c_in = model.list_layer[v_next.idx_layer]->c_in;
                        // int h_in = model.list_layer[v_next.idx_layer]->h_in;
                        // int w_in = model.list_layer[v_next.idx_layer]->w_in;
                        // checkCuda(cudaMemcpy(model.list_layer[v_next.idx_layer]->tempData, model.list_layer[v_next.idx_layer]->srcData,
                        //     n_in*c_in*h_in*w_in*sizeof(value_type), cudaMemcpyDeviceToDevice));
                    }
                }

                checkCUDNN(cudnnRNNForwardInference(cudnnHandle, current_index_layer->rnnDesc, 
                                                    1,
                                                    current_index_layer->xDesc,
                                                    current_index_layer->srcData,
                                                    current_index_layer->hxDesc,
                                                    current_index_layer->hx,
                                                    current_index_layer->cxDesc,
                                                    current_index_layer->cx,
                                                    current_index_layer->weightDesc,
                                                    current_index_layer->weight,
                                                    current_index_layer->yDesc,
                                                    current_index_layer->dstData,
                                                    current_index_layer->hyDesc,
                                                    current_index_layer->hy,
                                                    current_index_layer->cyDesc,
                                                    current_index_layer->cy,
                                                    current_index_layer->workSpace,
                                                    current_index_layer->sizeInBytes));
                double now = get_current_time(start);
                printf("Passed index %d at time %lfms\n\n", v_now.idx_layer, now);
                
                vector<int>::iterator iter=v_now.idx_request.begin();
                
                bool req_passed = false;
                for(; iter!=v_now.idx_request.end(); iter++){
                    if(request_current_unroll.at(*iter) < request_unroll.at(*iter) -1){
                        request_current_unroll.at(*iter) += 1;
                        req_passed = true;
                    }
                }

                /* need to separate passed req and not-passed req in the vector */
                if(req_passed > 0){
                    v_layer.push_back(v_now);
                }
                if(req_passed==0){
                    for(iter = v_now.idx_request.begin(); iter != v_now.idx_request.end(); iter++){
                        request_current_unroll.at(*iter) = 0;
                    }

                    v_now.idx_layer += 1;
                    if(v_now.idx_layer >= MAX_LAYER_NUM){
                        int request_done = v_now.idx_request.size();
                        model.data_first += request_done;
                        model.data_num -= request_done;

                        if(input_index == request_number) break;
                    }
                    else{
                        v_layer.push_back(v_now);
                        if(merge_request(&v_layer)){
                            printf("merged!\n");
                        }
                    }
                }
            }
        }
    }

  for(int i=0; i<MAX_LAYER_NUM; i++){
      cudaFree(model.list_layer[i]->workSpace);
  }
  return 0;
}
