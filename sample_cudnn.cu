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

#include <unistd.h>
#include <time.h>
#include <pthread.h>


#ifdef USE_CPP_11
#include <thread>
#endif

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

#define value_type float
#define DATA_PRECISION  CUDNN_DATA_FLOAT


/*
typedef enum
{
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6
} cudnnConvolutionFwdAlgo_t;
*/
int main(int argc, char **argv)
{
	// first cmdline parameter is used to specify which CNN algorithm to use for derivation
	cudnnConvolutionFwdAlgo_t fwdAlgo	= (cudnnConvolutionFwdAlgo_t)atoi(argv[1]);
	
	// create streams
	cudaStream_t myStream;
	cudaStreamCreate(&myStream);

	// cudnn handle
  cudnnHandle_t         cudnnHandle;  
  checkCUDNN(cudnnCreate(&cudnnHandle));

  // link cudnnHandle with this stream
  checkCUDNN(cudnnSetStream(cudnnHandle, myStream));

	//---------------
	// 1. input fmap
	//---------------
	// NCHW spec for input feature map (fmap) 
	int n_in	= 64;
	int c_in	= 64;
	int h_in	= 224;
	int w_in	= 224;
	// tensor descriptor for input fmap
  cudnnTensorDescriptor_t       srcTensorDesc;
  checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
  checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_in, c_in, h_in, w_in));
	// malloc input fmap data array
	void*     srcData;
  cudaMalloc(&srcData, n_in*c_in*h_in*w_in*sizeof(value_type));

	//---------------
	// 2-a. filters
	//---------------
	// declare tensor descriptors for CNN layer filters
	int k			= 128;
	int r			= 3;
	int s			= 3;
	// tensor descriptor for filters
  cudnnFilterDescriptor_t       filterDesc;
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c_in, r, s));
	// malloc filter data array 
  void*     filterData;
  cudaMalloc(&filterData, k*c_in*r*s*sizeof(value_type));

	//---------------
	// 2-b. Conv layer spec
	//---------------
	// declare descriptor for the convolution operation to be done within the layer we're testing here
	int pad_h	= 1;
	int pad_w	= 1;
	int stride_h	= 1;
	int stride_w	= 1;
  cudnnConvolutionDescriptor_t  convDesc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_w, stride_h, 1, 1, CUDNN_CROSS_CORRELATION));

	//---------------
	// 3. output fmap
	//---------------
	// NCHW spec for output feature map (fmap)
	// find dimension of convolution output
	int n_out, c_out, h_out, w_out;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));
	// declare tensor descriptor for input and output activation maps
  cudnnTensorDescriptor_t       dstTensorDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
  checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_out, c_out, h_out, w_out));
	// malloc output fmap data array
	void*			dstData;
  cudaMalloc(&dstData, n_out*c_out*h_out*w_out*sizeof(value_type));

	// allocate workspace if required
	std::cout<<"\n-----------------------\n1. fmap and filter size\n-----------------------"<<std::endl;
	std::cout<<" Input  Fmap size: "<<n_in*c_in*h_in*w_in*sizeof(value_type)<<" (bytes)"<<std::endl;
	std::cout<<" Output Fmap size: "<<n_out*c_out*h_out*w_out*sizeof(value_type)<<" (bytes)"<<std::endl;
	std::cout<<" Filter size: "<<k*c_in*r*s*sizeof(value_type)<<" (bytes)\n"<<std::endl;



	// test which cudnn algorithm 
  int requestedAlgoCount = 6; 
  int returnedAlgoCount;

	// profile results
  cudnnConvolutionFwdAlgoPerf_t*        fwdProfileResults;
  fwdProfileResults = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
  cudnnConvolutionFwdAlgoPerf_t *results = fwdProfileResults;

 
	std::cout<<"\n-----------------------\n2. Profile all algorithm\n-----------------------"<<std::endl;
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, requestedAlgoCount, &returnedAlgoCount, results)); 
  for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex) {
      printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, (unsigned long long)results[algoIndex].memory);
	}
	std::cout<<std::endl;

	std::cout<<"\n-----------------------"<<std::endl;
	printf("Chosen Algorithm: (%d) ",fwdAlgo);
	switch(fwdAlgo) {
		case 0:
			printf("IMPLCIT_GEMM\n");
			break;
		case 1:
			printf("IMPLCIT_PRECOMP_GEMM\n");
			break;
		case 2:
			printf("GEMM\n");
			break;
		case 3:
			printf("DIRECT\n");
			break;
		case 4:
			printf("FFT\n");
			break;
		case 5:
			printf("FFT_TILING\n");
			break;
		case 6:
			printf("WINOGRAD\n");
			break;
		default:
			printf("Invalid algorithm (ERROR)\n");
			assert(0);
	}
	std::cout<<"-----------------------"<<std::endl;

	//---------------
	// 4. Workspace
	//---------------
  // check if workspace is needed for fwdprop
  void* workSpace=NULL;
  size_t sizeInBytes=0;
  // derive workspace size
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                            srcTensorDesc,
                                            filterDesc,
                                            convDesc,
                                            dstTensorDesc,
																						fwdAlgo,
                                            &sizeInBytes));
	std::cout<<"[Note] WorkingSpace required: "<<sizeInBytes<<" (bytes)"<<std::endl;
	if(sizeInBytes!=0) {
		cudaMalloc(&workSpace, sizeInBytes);
	}

	// TIMER
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	// Allocate CUDA events that we'll use for timing
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// warm-up
  value_type alpha = value_type(1);
  value_type beta  = value_type(0);
	checkCUDNN(cudnnConvolutionForward(  cudnnHandle,
                            &alpha,
                            srcTensorDesc,
                            srcData,
                            filterDesc,
                            filterData,
                            convDesc,
														fwdAlgo,
                            workSpace,
                            sizeInBytes,
                            &beta,
                            dstTensorDesc,
                            dstData
                          ));


  // Record the start event
  checkCudaErrors(cudaEventRecord(start, NULL));


	// do forward propagation (conv layer)
  int nIter = 10;
  for (int j = 0; j < nIter; j++)
  {
		checkCUDNN(cudnnConvolutionForward(  cudnnHandle,
                            &alpha,
                            srcTensorDesc,
                            srcData,
                            filterDesc,
                            filterData,
                            convDesc,
														fwdAlgo,	
                            workSpace,
                            sizeInBytes,
                            &beta,
                            dstTensorDesc,
                            dstData
                          ));
	}
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, NULL));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

	// free memory for workspace
	if(sizeInBytes!=0) {
		cudaFree(workSpace);
	}

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  // Compute and print the performance
  float msecPerFwdprop = msecTotal / nIter;
  printf("(CONV layer) Time taken = %.3f (msec)\n",msecPerFwdprop);


	// do forward propagation (activation layer)
  cudnnActivationDescriptor_t   activationDesc;
  checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
  checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 1.0));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, NULL));
	nIter = 10;
  for (int j = 0; j < nIter; j++)
  {
		checkCUDNN(cudnnActivationForward(  cudnnHandle,
                                      activationDesc,
                                      &alpha,
                                      dstTensorDesc,
                                      dstData,
                                      &beta,
                                      dstTensorDesc,
                                      dstData
                                    ));
	}
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, NULL));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  // Compute and print the performance
  msecPerFwdprop = msecTotal / nIter;
  printf("(RELU layer) Time taken = %.3f (msec)\n",msecPerFwdprop);

  return 0;
}
