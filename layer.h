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

// #include "model.h"

#ifdef USE_CPP_11
#include <thread>
#endif

#define value_type float
#define DATA_PRECISION CUDNN_DATA_FLOAT

#define MODE_DYNAMIC 0
#define DYNAMIC_TERM 100

#define MODEL_TYPE 1
#define MAX_LAYER_NUM 52
/*  9 for MODEL_TYPE 2, 52 for MODEL_TYPE 1 */
#define DEADLINE 1000

#define MAX_BATCH_SIZE 32
#define MAX_REQ_SIZE 32

#define HIDDEN_SIZE_GNMT 1024

#define DEC_SEQ_LENGTH 1024

// #define MAX_LAYER_NUM 52

enum layer_t
{
    CONV_FIRST      = 0,
    CONV            = 1,
    CONV_LAST       = 2,
    RELU            = 3,
    FC              = 4,
    RNN             = 5,
    RNN_LAST        = 6,
    POOL_AVERAGE    = 7,
    POOL            = 8,
    CONV_RESIDUAL   = 9,
    SOFTMAX         = 10,
    ACTIVATE        = 11,

    ATTENTION       = 12,
    RNN_DECODER     = 13,
    NUM_LAYER_TYPES = 14
};

enum model_t
{
    DEFAULT         = 0,
    RESNET          = 1

};

class Layer{
public:
    Layer(
        enum layer_t         _layerType,
        cublasHandle_t* _cublasHandle,
        cudnnHandle_t*  _cudnnHandle,
        cudaStream_t*   _stream_compute,
        cudaStream_t*   _stream_memory,
        int _n, int _c, int _h, int _w,
        int _pad_h, int _pad_w, int _stride_h, int _stride_w,
        int _k, int _r, int _s,
        int _n_out, int _c_out, int _h_out, int _w_out,
        int _seqlength, int _hidden_size, int _num_layers, int idx);

    ~Layer(){};

    // CONV layer
    size_t  convolutionForward(bool* _offloaded, value_type** _offloadedSrcData_h);
    size_t  convolutionBackward(int _layer_id_to_prefetch, void* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, size_t* _memUsedFilter, size_t* _memUsedData);

    // FC_GEMM layer
    size_t  fullyConnectedForward(bool* _offloaded, value_type** _offloadedSrcData_h);
    size_t  fullyConnectedBackward(int _layer_id_to_prefetch, void* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, size_t* _memUsedFilter, size_t* _memUsedData);

    // RELU layer
    size_t  activationForward(bool* _offloaded, value_type** _offloadedSrcData_h);
    size_t  activationBackward(int _layer_id_to_prefetch, void* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes);

    // gets
    enum layer_t       LayerType()   { assert(layerType<NUM_LAYER_TYPES); return layerType; }
    void*         SrcData()     { return srcData;                 }
    void*         DstData()     { return dstData;                 }
    void*         GradData()    { return gradData;                }
    int           SrcDataSize() { return n_in*c_in*h_in*w_in;     }
    int           DstDataSize() { return n_out*c_out*h_out*w_out; }
    int           FilterDataSize()  { return k*c_in*r*s;          }

    void          setDstData(void* point)     { this->dstData = point; }
    void          setContextData(void* point)   { this->contextData = point;}
    // mem usage
    size_t        memUsageSrcData();
    size_t        memUsageDstData();
    size_t        memUsageFilterData();
    size_t        memUsageDiffData();
    size_t        memUsageGradData();
    size_t        memUsageWorkSpaceFwd();
    size_t        memUsageWorkSpaceBwd();

    size_t        deriveMemUsageFwd();
    size_t        deriveMemUsageBwd();
    size_t        deriveMemUsageFwdLayerWise();
    size_t        deriveMemUsageBwdLayerWise();

    void          decrementRefCntFwd()  { assert(refCntFwd>0);  refCntFwd--;  }
    void          decrementRefCntBwd()  { assert(refCntBwd>0);  refCntBwd--;  }

    void          change_size_RNN(int initialized, int size);
    // private:
    enum layer_t   layerType;
    int       id;
    int type;
    //--------
    // input
    //--------
    // input NCHW
    int       n_in;
    int       c_in;
    int       h_in;
    int       w_in;
    // padding, stride
    int       pad_h;
    int       pad_w;
    int       stride_h;
    int       stride_w;
    int       k;
    int       r;
    int       s;
    //--------
    // output
    //--------
    // output NCHW
    int       n_out;
    int       c_out;
    int       h_out;
    int       w_out;

    /* RNN params */

    int       seqlength;
    int       hidden_size;
    int       num_layers;
    int       unrolled_num[MAX_LAYER_NUM];
    // int       direction;
    int idx;

    void *hx;
    void *cx;

    void *hy;
    void *cy;

    void* weight;
    void* workSpace;
    size_t sizeInBytes;

    // streams
    cudaStream_t*                 stream_compute;
    cudaStream_t*                 stream_memory;

    //------------
    // ptrs to memory
    //------------
    void*     srcData;
    void*     filterData;
    void*     tempData;

    void*     biasData;
    void*     dstData;
    void*     diffData;
    void*     gradData;
    void*     gemmData;
    void*     weightData;
    void*     softmaxData;
    void*     contextData;

    // for backprop
    //value_type*   diffData; // backprop input to layer
    //value_type*   gradData; // backprop output of layer for 1) backpropData (when forwarding to previous layer) or 2) backpropFilter (to update wGrad within this layer)

    //--------
    // cuDNN 
    //--------

    cudnnHandle_t*                cudnnHandle;  // shared across all layers
    cudnnDataType_t               dataType;
    cudnnConvolutionMode_t        modeConv;
    cudnnTensorFormat_t           tensorFormat;
    cudnnTensorDescriptor_t       srcTensorDesc, dstTensorDesc, biasTensorDesc, gemmTensorDesc, weightTensorDesc, softmaxTensorDesc, contextTensorDesc;
    cudnnTensorDescriptor_t       *xDesc;
    cudnnTensorDescriptor_t       *yDesc;
    cudnnTensorDescriptor_t       hxDesc, cxDesc, hyDesc, cyDesc;
    cudnnFilterDescriptor_t       weightDesc;
    cudnnTensorDescriptor_t       srcDiffTensorDesc, dstDiffTensorDesc;   // backprop
    cudnnFilterDescriptor_t       filterDesc;
    cudnnConvolutionDescriptor_t  convDesc;
    cudnnRNNDescriptor_t          rnnDesc;
    cudnnPoolingDescriptor_t      poolingDesc;
    bool                          inPlaceOp;
    cudnnConvolutionFwdAlgo_t     fwdAlgo;
    int algo_int;

    cudnnDropoutDescriptor_t      dropoutDesc;
    cudnnRNNInputMode_t           inputmodeRNN;
    cudnnRNNMode_t                modeRNN;
    cudnnDirectionMode_t          direction;
    cudnnRNNAlgo_t                rnnAlgo;

    cudnnSoftmaxAlgorithm_t       softmaxAlgo;
    cudnnSoftmaxMode_t            softmaxMode;
    cudnnActivationDescriptor_t   activationDesc;
    
    int rnnalgo_int;

    size_t weightsSize;
    size_t stateSize;
    void* states;

    //--------
    // cuBLAS
    //--------
    cublasHandle_t*               cublasHandle;
    //cublasOperation_t             cublasOp;

    // GOOGLENET
    bool  insideInceptionModule;
    bool  headOfFork;
    bool  tailOfFork;
    int   forkId;
    int   concatenateChannelOffset;
    std::vector<int>  producerLayerId;
    std::vector<int>  consumerLayerId;

    int   refCntFwd;
    int   refCntBwd;

    int status;

    friend struct Network;
};

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
