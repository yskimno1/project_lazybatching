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

using namespace std;

enum layer_t model_info[] = {CONV_LAST};

struct Model_layer createModel(int type, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, cudaStream_t myStream_compute, cudaStream_t myStream_mem){
    struct Model_layer model;
    int hidden_size = HIDDEN_SIZE_GNMT;
    std::cout<<"Model Type: "<<type<<std::endl;

    if(type == 1){ model = create_Resnet(cublasHandle, cudnnHandle, myStream_compute, myStream_mem); }
    else if(type == 2){
        model = create_GNMT(cublasHandle, cudnnHandle, myStream_compute, myStream_mem);

        cudaMalloc(&model.matrix_W1,hidden_size * hidden_size * sizeof(value_type) );
        cudaMalloc(&model.matrix_W2, hidden_size * hidden_size * sizeof(value_type));
        cudaMalloc(&model.vector_score, hidden_size * sizeof(value_type));
    }
    else{
        for(int i=0; i<MAX_LAYER_NUM; i++){
            model.list_layer[i] = new Layer(model_info[i], &cublasHandle, &cudnnHandle, &myStream_compute, &myStream_mem,
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

struct vector_layer set_vector(int idx_layer, int index_request, int seqlength){
    struct vector_layer new_vector;
    new_vector.idx_layer = idx_layer;
    new_vector.idx_request.push_back(index_request);
    new_vector.seqlength = seqlength;
    return new_vector;
}

bool merge_request(vector<struct vector_layer>* v_layer){
    int idx_end = (*v_layer).size()-1;
    if(idx_end <= 0) return false;
    struct vector_layer v_1 = (*v_layer)[idx_end];
    struct vector_layer v_2 = (*v_layer)[idx_end-1];
    if(v_1.idx_layer == v_2.idx_layer){
        std::cout<<"\n                 before merge, request size: "<<v_2.idx_request.size()<<", current size: "<<(*v_layer).size()<<std::endl;
        v_2.idx_request.insert(v_2.idx_request.end(), v_1.idx_request.begin(), v_1.idx_request.end());
        (*v_layer).pop_back();
        (*v_layer).pop_back();
        (*v_layer).push_back(v_2);
        std::cout<<"                 after merge, request size : "<<v_2.idx_request.size()<<", current size: "<<(*v_layer).size()<<std::endl;

        return true;
    }
    else return false;
}

int main(int argc, char **argv)
{
    vector<float> vector_request;
    vector<double> latency_request;
    vector<int> unroll_request;
    vector<int> current_unroll_request;
    int total_request_num;
    int fixed_seq_length = 5;
    
    ifstream filename(argv[1]);
    char buf[100];
    if(filename.is_open()){
        while(filename){
            filename.getline(buf, 100, ',');
            if(strlen(buf)) vector_request.push_back(atof(buf));
        }
        total_request_num = vector_request.size();
        std::cout<<"Number of requests: "<<total_request_num<<std::endl;
        for(int i=0; i<total_request_num; i++){
            unroll_request.push_back(fixed_seq_length);
            current_unroll_request.push_back(0);
            latency_request.push_back(-vector_request.at(i));
        }
    }
    else exit(-1);

    /* ---------------------------- setup ---------------------------- */

    cudaStream_t myStream_compute;
    cudaStreamCreate(&myStream_compute);
    cudaStream_t myStream_mem;
    cudaStreamCreate(&myStream_mem);

	// cudnn handle
    cudnnHandle_t         cudnnHandle;
    cublasHandle_t        cublasHandle;
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUBLAS(cublasCreate(&cublasHandle));

    // link cudnnHandle with this stream
    checkCUDNN(cudnnSetStream(cudnnHandle, myStream_compute));
    checkCUDNN(cudnnSetStream(cudnnHandle, myStream_mem));

    vector<struct vector_layer> v_layer;
    value_type alpha = value_type(1);
    value_type beta  = value_type(0);

    struct Model_layer model = createModel(MODEL_TYPE, cublasHandle, cudnnHandle, myStream_compute, myStream_mem);

    clock_t start;
    start = clock();
    bool stall_request;
    int index_request = 0;
    while(index_request <= total_request_num){
        
        stall_request = false;
        if(model.data_num >= MAX_REQ_SIZE)    stall_request = true;
    
        double time_request_started = get_current_time(start);
        if(stall_request==false && index_request != total_request_num &&
                    time_request_started > vector_request.at(index_request)){ 
                    
            struct vector_layer v_new = set_vector(0, index_request, unroll_request.at(index_request));
            v_layer.push_back(v_new);
            index_request += 1;
            model.data_num += 1;
            std::cout<<"                       --came at time : "<<time_request_started<<"--"<<std::endl;
            std::cout<<"                       --input index : "<<index_request<<"--"<<std::endl;
            if(merge_request(&v_layer))               std::cout<<"                 merged!\n"<<std::endl;
        }
        else{
            if(v_layer.size() <= 0) continue;
            std::cout<<"current req num: "<<model.data_num<<", starts at "<<model.data_first<<std::endl;

            int idx_end = v_layer.size()-1;
            struct vector_layer v_now = v_layer[idx_end];
            v_layer.pop_back();

            Layer* current_index_layer = model.list_layer[v_now.idx_layer];

            if(MODEL_TYPE==2){
                switch (current_index_layer->layerType){
                    case(ATTENTION):
                        {
                            std::cout<<"Attention Layer"<<std::endl;
                            int num_req = v_now.idx_request.size();
                            
                            
                            
                            /* needs descriptor change */
        
        
        
                            for(int i=0; i<current_index_layer->n_in/MAX_BATCH_SIZE*num_req; i++){
                                checkCUBLAS(cublasSgemm(cublasHandle,  CUBLAS_OP_N, CUBLAS_OP_N,
                                    current_index_layer->hidden_size, current_index_layer->seqlength, current_index_layer->hidden_size,
                                    &alpha,
                                    (float *)(model.matrix_W1), current_index_layer->hidden_size,
                                    ((float *)current_index_layer->srcData + i * current_index_layer->hidden_size*current_index_layer->seqlength*sizeof(value_type)), current_index_layer->hidden_size,
                                    &beta,
                                    ((float *)current_index_layer->gemmData + i * current_index_layer->hidden_size*current_index_layer->seqlength*sizeof(value_type)), current_index_layer->hidden_size
                                ));
                            }
        
                            checkCUDNN(cudnnActivationForward(cudnnHandle, current_index_layer->activationDesc, &alpha,
                                current_index_layer->gemmTensorDesc,
                                current_index_layer->gemmData, 
                                &beta,
                                current_index_layer->dstTensorDesc,
                                current_index_layer->dstData
                            ));
        
                            for(int i=0; i<current_index_layer->n_in/MAX_BATCH_SIZE*num_req; i++){
                                checkCUBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                    current_index_layer->hidden_size, current_index_layer->seqlength,
                                    &alpha,
                                    ((float *)current_index_layer->dstData + i * current_index_layer->hidden_size*current_index_layer->seqlength*sizeof(value_type)), current_index_layer->hidden_size,
                                    (float *)model.vector_score, 1,
                                    &beta,
                                    ((float *)current_index_layer->weightData + i * 1 * current_index_layer->seqlength*sizeof(value_type)), 1
                                ));
                            }
        
                            checkCUDNN(cudnnSoftmaxForward(cudnnHandle, current_index_layer->softmaxAlgo, current_index_layer->softmaxMode,
                                &alpha,
                                current_index_layer->weightTensorDesc,
                                current_index_layer->weightData,
                                &beta,
                                current_index_layer->softmaxTensorDesc,
                                current_index_layer->softmaxData
                            ));
        
                            for(int i=0; i<current_index_layer->n_in/MAX_BATCH_SIZE*num_req; i++){
                                checkCUBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                    current_index_layer->hidden_size, current_index_layer->seqlength,
                                    &alpha,
                                    ((float *)current_index_layer->srcData + i * current_index_layer->hidden_size*current_index_layer->seqlength*sizeof(value_type)), current_index_layer->hidden_size,
                                    (float *)current_index_layer->softmaxData + i * 1 * current_index_layer->seqlength*sizeof(value_type), 1,
                                    &beta,
                                    ((float *)current_index_layer->contextData + i * 1 * current_index_layer->hidden_size * sizeof(value_type)), 1
                                ));
                            }
        
                            double now = get_current_time(start);
                            std::cout<<"Passed index "<<v_now.idx_layer<<" at time "<<now<<"ms\n"<<std::endl;
        
                            checkCuda(cudaMemcpy(model.list_layer[v_now.idx_layer+1]->srcData, model.list_layer[v_now.idx_layer]->contextData,
                                current_index_layer->n_in/MAX_BATCH_SIZE*num_req*1*1*current_index_layer->hidden_size*sizeof(value_type), cudaMemcpyDeviceToDevice));
        
                            v_now.idx_layer += 1;
                            if(v_now.idx_layer >= MAX_LAYER_NUM){
                                int request_done = v_now.idx_request.size();
                                model.data_first += request_done;
                                model.data_num -= request_done;
            
                                vector<int>::iterator iter=v_now.idx_request.begin();
                            
                                for(; iter!=v_now.idx_request.end(); iter++){   
                                    latency_request.at(*iter) += now;
                                    printf("%lfms, latency of request no.%d : %lfms\n\n", now, *iter, latency_request.at(*iter));
                                }
                                if(index_request == total_request_num) exit(0);
                            }
                            else{
                                v_layer.push_back(v_now);
                                if(merge_request(&v_layer)){
                                    printf("merged!\n");
                                }
                            }
                        }
                    break;

                    case(RNN):
                    case(RNN_LAST):
                    case(RNN_DECODER):
                        {
                            int num_req = v_now.idx_request.size();
                        
                            current_index_layer->change_size_RNN(0, current_index_layer->n_in/MAX_BATCH_SIZE * num_req);
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
                            printf("Passed layer %d at time %lfms\n\n", v_now.idx_layer, now);
                            bool req_notyet = false;
                            vector<int>::iterator iter=v_now.idx_request.begin();
                            
                            if(current_index_layer->layerType != RNN_LAST){
                                v_now.idx_layer += 1;                  
                                v_layer.push_back(v_now);
                                if(merge_request(&v_layer)){
                                    printf("merged!\n");
                                }
                            }
                            else{
                                for(; iter!=v_now.idx_request.end(); iter++){
                                    if(current_unroll_request.at(*iter) < unroll_request.at(*iter)-1){
                                        current_unroll_request.at(*iter) += 1;
                                        req_notyet = true;
                                    }
                                }    
                                /* need to separate passed req and not-passed req filename the vector */
                                if(req_notyet > 0){
                                    v_now.idx_layer = 4;
                                    v_layer.push_back(v_now);
                                }
                                if(req_notyet==0){
                                    for(iter = v_now.idx_request.begin(); iter != v_now.idx_request.end(); iter++){
                                        current_unroll_request.at(*iter) = 0;
                                    }
        
                                    int request_done = v_now.idx_request.size();
                                    model.data_first += request_done;
                                    model.data_num -= request_done;
                                    
                                    printf("--------------------------------------------\n");
                                    for(iter=v_now.idx_request.begin(); iter!=v_now.idx_request.end(); iter++){
                                        latency_request.at(*iter) += now;
                                        printf("| %lfms, latency of request no.%d : %lfms |\n", now, *iter, latency_request.at(*iter));
                                    }
                                    printf("--------------------------------------------\n");
                                    
                                    if(index_request == total_request_num) exit(0);
                                }                
                            }
                        }
                    break;
                }
            }
            else if(MODEL_TYPE==1){
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
    
                            latency_request.at(*iter) += now;
                            printf("%lfms, latency of request no.%d : %lfms\n\n", now, *iter, latency_request.at(*iter));
                        }
    
                        if(index_request == total_request_num) exit(0);
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
    
                        if(index_request == total_request_num) exit(0);
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
                    
                    bool req_notyet = false;
                    for(; iter!=v_now.idx_request.end(); iter++){
                        if(current_unroll_request.at(*iter) < unroll_request.at(*iter)-1){
                            current_unroll_request.at(*iter) += 1;
                            req_notyet = true;
                        }
                    }
    
                    /* need to separate passed req and not-passed req filename the vector */
                    if(req_notyet > 0){
                        v_layer.push_back(v_now);
                    }
                    if(req_notyet==0){
                        for(iter = v_now.idx_request.begin(); iter != v_now.idx_request.end(); iter++){
                            current_unroll_request.at(*iter) = 0;
                        }
    
                        v_now.idx_layer += 1;
                        if(v_now.idx_layer >= MAX_LAYER_NUM){
                            int request_done = v_now.idx_request.size();
                            model.data_first += request_done;
                            model.data_num -= request_done;
    
                            if(index_request == total_request_num) break;
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
    }

  for(int i=0; i<MAX_LAYER_NUM; i++){
      cudaFree(model.list_layer[i]->workSpace);
  }
  return 0;
}
