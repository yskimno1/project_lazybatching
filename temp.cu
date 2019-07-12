#include <stdio.h>
#include <string.h>

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <assert.h>

#include <cudnn.h>
#include <cublas_v2.h>

#include <unistd.h>
#include <time.h>
#include <pthread.h>
// #include <cnmem.h>

#ifdef USE_CPP_11
#include <thread>
#endif

#include "layer.h"
#include "model.h"

using namespace std;

double input_time[] = {1, 1.1, 5, 6, 7};
enum layer_t model_info[] = {CONV, CONV, CONV, CONV, CONV};

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
    //assert(result == cudaSuccess);
  }

  return result;
}

Layer::Layer(
  enum layer_t         _layerType,
  cudnnHandle_t*  _cudnnHandle,
  cudaStream_t*   _stream_compute,
  cudaStream_t*   _stream_memory,
  int _id,
  int _n, int _c, int _h, int _w, 
  int _pad_h, int _pad_w, int _stride_h, int _stride_w, 
  int _k, int _r, int _s,
  int _n_out, int _c_out,
  int _h_out, int _w_out):  layerType(_layerType), 
                            cudnnHandle(_cudnnHandle), stream_compute(_stream_compute), stream_memory(_stream_memory), 
                            id(_id),
                            n_in(_n), c_in(_c), h_in(_h), w_in(_w),
                            pad_h(_pad_h), pad_w(_pad_w), stride_h(_stride_h), stride_w(_stride_w),
                            k(_k), r(_r), s(_s),
                            n_out(_n_out), c_out(_c_out), h_out(_h_out), w_out(_w_out)
{
  // GOOGLENET
  insideInceptionModule = false;
  headOfFork            = false;
  tailOfFork            = false;
  forkId                = -1;
  concatenateChannelOffset = -1;
  producerLayerId.clear();
  consumerLayerId.clear();
  refCntFwd             = 0;
  refCntBwd             = 0;


  // default    
  dataType      = DATA_PRECISION;
  //  modeConv      = CUDNN_CONVOLUTION;
  modeConv      = CUDNN_CROSS_CORRELATION;
  tensorFormat  = CUDNN_TENSOR_NCHW;
  inPlaceOp     = false; 
  // No srcData cudaMalloc() when initializing Layer() 
  srcData       = NULL;
  tempData      = NULL;
  filterData    = NULL;
  biasData      = NULL;
  dstData       = NULL;
  diffData      = NULL; // input to layer when backprop
  gradData      = NULL; // output of layer when backprop
  algo_int      = 4;
  fwdAlgo       = (cudnnConvolutionFwdAlgo_t) algo_int;

  /* RNN params */

  seqlength     = 1;
  hidden_size   = 100;
  num_layers    = 50;
  inputmodeRNN  = CUDNN_LINEAR_INPUT;
  modeRNN       = CUDNN_RNN_RELU;
  direction     = CUDNN_UNIDIRECTIONAL;

  rnnalgo_int   = 2;
  rnnAlgo       = CUDNN_RNN_ALGO_STANDARD;

  hx = NULL;
  cx = NULL;

  hy = NULL;
  cy = NULL;

  int n = n_in;

  switch(_layerType) {

  case FC:
    {

    }
  case RNN:
    {
        cudaMalloc(&srcData, seqlength * hidden_size * n * sizeof(value_type));
        cudaMalloc(&dstData, seqlength * hidden_size * n * sizeof(value_type));

        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc));

        checkCUDNN(cudnnCreateTensorDescriptor(&hxDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cxDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&hyDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cyDesc));

        // cudaMalloc(&x, seqlength * hidden_size * n * sizeof(value_type));        cudaMalloc(&hx, num_layers * hidden_size * n * sizeof(value_type));
        cudaMalloc(&cx, num_layers * hidden_size * n * sizeof(value_type));

        cudaMalloc(&hy, num_layers * hidden_size * n * sizeof(value_type));
        cudaMalloc(&cy, num_layers * hidden_size * n * sizeof(value_type));

        xDesc = (cudnnTensorDescriptor_t* )malloc(seqlength * sizeof(cudnnTensorDescriptor_t));
        yDesc = (cudnnTensorDescriptor_t* )malloc(seqlength * sizeof(cudnnTensorDescriptor_t));

        int dimA[3];
        int strideA[3];

        for(int i=0; i<seqlength; i++){
            checkCUDNN(cudnnCreateTensorDescriptor(&xDesc[i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&yDesc[i]));
            dimA[0] = n;
            dimA[1] = hidden_size;
            dimA[2] = 1;
            strideA[0] = dimA[2] * dimA[1];
            strideA[1] = dimA[2];
            strideA[2] = 1;

            cudnnSetTensorNdDescriptor(xDesc[i], dataType, 3, dimA, strideA);
            cudnnSetTensorNdDescriptor(yDesc[i], dataType, 3, dimA, strideA);
        }

        dimA[0] = num_layers;
        dimA[1] = n;
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

        initGPUData((float*)srcData, seqlength * hidden_size * n, 1.f);
        if (hx != NULL) initGPUData((float*)hx, num_layers * hidden_size * n, 1.f);
        if (cx != NULL) initGPUData((float*)cx, num_layers * hidden_size * n , 1.f);
     
        int numLinearLayers = 2;
        for(int layer=0; layer<num_layers; layer++){
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


    }
  case CONV:
    {
      // create cudnn descriptors
      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
      checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

      value_type *filterData_h	= (value_type*)malloc(k*c_in*r*s*sizeof(value_type));

      for(unsigned i=0; i<k*c_in*r*s; i++)	filterData_h[i]	= (value_type)i/(k*c_in*r*s);
      cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type));
      cudaMalloc(&dstData, n_out*c_out*h_out*w_out*sizeof(value_type));
      cudaMalloc(&filterData, k*c_in*r*s*sizeof(value_type));
      cudaMalloc(&tempData, n_in*c_in*h_in*w_in*sizeof(value_type));

      checkCuda(cudaMemcpy(filterData, filterData_h, k*c_in*r*s*sizeof(value_type), cudaMemcpyHostToDevice));
      // free memory
      free(filterData_h);

      // set input descriptors
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));
      checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat, k, c_in, r, s));
      checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_w, stride_h, 1, 1, modeConv, CUDNN_DATA_FLOAT));

      // find dimension of convolution output
      checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));

      // set output descriptor based on above
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));

    }
    break;
  case CONV_LAST:
    {
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  
        value_type *filterData_h	= (value_type*)malloc(k*c_in*r*s*sizeof(value_type));
  
        for(unsigned i=0; i<k*c_in*r*s; i++)	filterData_h[i]	= (value_type)i/(k*c_in*r*s);
        cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type));
        cudaMalloc(&dstData, n_out*c_out*h_out*w_out*sizeof(value_type));
        cudaMalloc(&filterData, k*c_in*r*s*sizeof(value_type));
        cudaMalloc(&tempData, n_in*c_in*h_in*w_in*sizeof(value_type));
  
        checkCuda(cudaMemcpy(filterData, filterData_h, k*c_in*r*s*sizeof(value_type), cudaMemcpyHostToDevice));
        // free memory
        free(filterData_h);
  
        // set input descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat, k, c_in, r, s));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_w, stride_h, 1, 1, modeConv, CUDNN_DATA_FLOAT));
  
        // find dimension of convolution output
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));
  
        // set output descriptor based on above
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
    }
  case RELU:
      // create cudnn descriptors
      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));

      // initialize n_/c_/h_/w_out
      n_out = n_in;
      c_out = c_in;
      h_out = h_in;
      w_out = w_in;

      // set input descriptors
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, tensorFormat, dataType, n_in, c_in, h_in, w_in));

      // set output descriptor based on above
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, tensorFormat, dataType, n_out, c_out, h_out, w_out));
      // default behavior of RELU is to do in-place op
      this->inPlaceOp = true;
    break;
  default:
    assert(0);
    break;
  }

  #ifdef _DEBUG_
  printf("Layer-%2d definition (Type=%d)\n", id, layerType);
  printf("Input:  %4d %4d %4d %4d\n", n_in, c_in, h_in, w_in);
  printf("Output: %4d %4d %4d %4d\n\n", n_out, c_out, h_out, w_out);
  #endif
  // create descriptor for backprop
  checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}

size_t Layer::convolutionForward(bool* _offloaded, value_type** _offloadedSrcData_h) {
  // sanity check: make sure all input/filter/output memory is allocated
  assert(srcData!=NULL);
  assert(filterData!=NULL);
  assert(dstData!=NULL);

  return 0;
}

struct Model_layer createModel(int type, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem, 
                    int n_in, int c_in, int h_in, int w_in, int pad_h, int pad_w, int stride_h, int stride_w, 
                    int k, int r, int s, int n_out, int c_out, int h_out, int w_out){
    struct Model_layer model;
    
    if(type == 0){
        model.list_layer[0] = new Layer(model_info[0], &cudnnHandle, &myStream_compute, &myStream_mem, 0, n_in, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, k, r, s, n_out, c_out, h_out, w_out);
        for(int i=1; i<MAX_LAYER_NUM-1; i++){
            model.list_layer[i] = new Layer(model_info[i], &cudnnHandle, &myStream_compute, &myStream_mem, i, n_in, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, k, r, s, n_out, c_out, h_out, w_out);
            model.list_layer[i-1]->setDstData(model.list_layer[i]->SrcData());
        }
        printf("max num :%d\n", MAX_LAYER_NUM-1);
        model.list_layer[MAX_LAYER_NUM-1] = new Layer(model_info[MAX_LAYER_NUM-1], &cudnnHandle, &myStream_compute, &myStream_mem, MAX_LAYER_NUM-1, n_in, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, k, r, s, n_out, c_out, h_out, w_out);
        model.list_layer[MAX_LAYER_NUM-2]->setDstData(model.list_layer[MAX_LAYER_NUM-1]->SrcData());

        model.current_mode = DEFAULT_STATE;
        model.current_index = 0;
        model.global_index = 0;
        model.data_exist = 0;
        return model;
    }
    else if(type == 1){
        model.list_layer[0] = new Layer(RNN, &cudnnHandle, &myStream_compute, &myStream_mem, 0, n_in, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, k, r, s, n_out, c_out, h_out, w_out);
        model.current_mode = DEFAULT_STATE;
        model.current_index = 0;
        model.global_index = 0;
        model.data_exist = 0;
        return model;
    }
    return model;
}

struct Model_layer changeModel(struct Model_layer model, int index, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem,
    int n_in, int c_in, int h_in, int w_in, int pad_h, int pad_w, int stride_h, int stride_w, 
    int k, int r, int s, int n_out, int c_out, int h_out, int w_out){

    int i;
    for(i=index; i<MAX_LAYER_NUM-1; i++){
        Layer* temp_layer = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem, 0, n_in, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, k, r, s, n_out, c_out, h_out, w_out);    
        if(i==index){
            size_t temp_size = model.list_layer[index]->n_in * model.list_layer[index]->c_in * model.list_layer[index]->h_in * model.list_layer[index]->w_in * sizeof(value_type);
            checkCuda(cudaMemcpy(temp_layer->srcData, model.list_layer[index]->tempData, temp_size, cudaMemcpyDeviceToDevice));

            checkCuda(cudaMemcpy( (void *)((int *)(temp_layer->srcData) + temp_size), model.list_layer[index]->srcData, n_in*c_in*h_in*w_in*sizeof(value_type) - temp_size, cudaMemcpyDeviceToDevice));
        }
        cudaFree(model.list_layer[i]->srcData);
        cudaFree(model.list_layer[i]->dstData);
        cudaFree(model.list_layer[i]->filterData);
        delete model.list_layer[i];

        model.list_layer[i] = temp_layer;
        model.list_layer[i-1]->setDstData(model.list_layer[i]->SrcData());
    }

    cudaFree(model.list_layer[MAX_LAYER_NUM-1]->srcData);
    cudaFree(model.list_layer[MAX_LAYER_NUM-1]->filterData);
    cudaFree(model.list_layer[MAX_LAYER_NUM-1]->dstData);
    delete model.list_layer[MAX_LAYER_NUM-1];

    model.list_layer[MAX_LAYER_NUM-1] = new Layer(CONV, &cudnnHandle, &myStream_compute, &myStream_mem, 0, n_in, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, k, r, s, n_out, c_out, h_out, w_out);
    model.list_layer[MAX_LAYER_NUM-2]->setDstData(model.list_layer[MAX_LAYER_NUM-1]->SrcData());

    return model;
}

bool make_delay(void){
    return true;
}

double get_current_time(clock_t start){
    clock_t now = clock();
    return (double)(now-start)/ CLOCKS_PER_SEC;
}

struct vector_layer{
    int idx_layer;
    vector<int> idx_request;
};

struct vector_layer set_vector(int idx_layer, int input_index){
    struct vector_layer new_vector;
    new_vector.idx_layer = idx_layer;
    new_vector.idx_request.push_back(input_index);
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
        printf("after merge, request size : %d, current size - %d\n", v_2.idx_request.size(), (*v_layer).size());

        (*v_layer).push_back(v_2);
        return true;
    }
    else return false;
}

int main(int argc, char **argv)
{

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

	// NCHW spec for input feature map (fmap) 
	int n_in	= 1;
	int c_in	= 64;
	int h_in	= 224;
    int w_in	= 224;

	int k			= 64;
	int r			= 3;
	int s			= 3;

    int pad_h	= 1;
	int pad_w	= 1;
	int stride_h	= 1;
    int stride_w	= 1;
  
    int n_out	= 1;
	int c_out	= 64;
	int h_out	= 224;
    int w_out	= 224;

    vector<struct vector_layer> v_layer;

    struct Model_layer model = createModel(0, cudnnHandle, myStream_compute, myStream_mem, 
        n_in, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, 
        k, r, s, n_out, c_out, h_out, w_out);

    value_type alpha = value_type(1);
    value_type beta  = value_type(0);
 
    start = clock();
    int num_input = sizeof(input_time)/sizeof(input_time[0]);

    while(input_index <= num_input){
        double now = get_current_time(start);
        if(input_index != num_input && now > input_time[input_index]){ /* New input condition */
            struct vector_layer v_new = set_vector(0, input_index);
            
            v_layer.push_back(v_new);
            input_index += 1;
            printf("came at time : %lf\n", now);

            if(merge_request(&v_layer)){
                int index = v_layer[v_layer.size()-1].idx_layer;
                int n_new = v_layer[v_layer.size()-1].idx_request.size();
                printf("request size  : %d\n", n_new);
                model = changeModel(model, index, cudnnHandle, myStream_compute, myStream_mem,
                    n_new, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, 
                    k, r, s, n_out, c_out, h_out, w_out);
            }
        }

        else{
            if(v_layer.size() <= 0) continue;
            
            int idx_end = v_layer.size()-1;
            struct vector_layer v_now = v_layer[idx_end];

            v_layer.pop_back();
            void* workSpace=NULL;
            size_t sizeInBytes=0;
            Layer* current_index_layer = model.list_layer[v_now.idx_layer];
            // derive workspace size
            if(current_index_layer->layerType == CONV || current_index_layer->layerType == CONV_LAST){
                checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                        current_index_layer->srcTensorDesc,
                                                        current_index_layer->filterDesc,
                                                        current_index_layer->convDesc,
                                                        current_index_layer->dstTensorDesc,
                                                        current_index_layer->fwdAlgo,
                                                        &sizeInBytes));
                std::cout<<"[Note] RNN WorkingSpace required: "<<sizeInBytes<<" (bytes)"<<std::endl;
                if(sizeInBytes!=0)   cudaMalloc(&workSpace, sizeInBytes);

                // bool save_temp = false;
                
                if(idx_end>0){
                    struct vector_layer v_next = v_layer[idx_end-1];
                    if(v_next.idx_layer == v_now.idx_layer +1){
                        // save_temp = true;
                        std::cout<<"save temp!" <<std::endl;
                        int n_in = model.list_layer[v_next.idx_layer]->n_in;
                        int c_in = model.list_layer[v_next.idx_layer]->c_in;
                        int h_in = model.list_layer[v_next.idx_layer]->h_in;
                        int w_in = model.list_layer[v_next.idx_layer]->w_in;
        
                        checkCuda(cudaMemcpy(model.list_layer[v_next.idx_layer]->tempData, model.list_layer[v_next.idx_layer]->srcData,
                            n_in*c_in*h_in*w_in*sizeof(value_type), cudaMemcpyDeviceToDevice));
                    }
                }

                checkCUDNN(cudnnConvolutionForward( cudnnHandle, &alpha, current_index_layer->srcTensorDesc,
                                                    current_index_layer->srcData,
                                                    current_index_layer->filterDesc,
                                                    current_index_layer->filterData,
                                                    current_index_layer->convDesc,
                                                    current_index_layer->fwdAlgo,
                                                    workSpace,
                                                    sizeInBytes,
                                                    &beta,
                                                    current_index_layer->dstTensorDesc,
                                                    current_index_layer->dstData
                                                ));
            }
            else{
                checkCUDNN(cudnnGetRNNWorkspaceSize(cudnnHandle, current_index_layer->rnnDesc, current_index_layer->seqlength, current_index_layer->xDesc, &sizeInBytes));
                std::cout<<"[Note] WorkingSpace required: "<<sizeInBytes<<" (bytes)"<<std::endl;

                if(sizeInBytes!=0)   cudaMalloc(&workSpace, sizeInBytes);

                checkCUDNN(cudnnRNNForwardInference(cudnnHandle, current_index_layer->rnnDesc, 
                                                    current_index_layer->seqlength,
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
                                                    workSpace,
                                                    sizeInBytes));
            }
            
            printf("passed index %d at time : %lf\n", v_now.idx_layer, now);
            v_now.idx_layer += 1;
            
            if(v_now.idx_layer < MAX_LAYER_NUM){
                v_layer.push_back(v_now);
                if(merge_request(&v_layer)){
                    printf("merged!\n");
                    int index = v_layer[v_layer.size()-1].idx_layer;
                    int n_new = v_layer[v_layer.size()-1].idx_request.size();
                    printf("request size  : %d\n", n_new);
                    model = changeModel(model, index, cudnnHandle, myStream_compute, myStream_mem,
                        n_new, c_in, h_in, w_in, pad_h, pad_w, stride_h, stride_w, 
                        k, r, s, n_out, c_out, h_out, w_out);
                }
            }
            else{
                if(input_index == num_input) break;
            }
            if(sizeInBytes !=0) cudaFree(workSpace);
        }
    }

  return 0;
}