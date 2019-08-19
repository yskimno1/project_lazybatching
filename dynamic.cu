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
            request_unroll.push_back(50);
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
    cublasHandle_t        cublasHandle;

    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUBLAS(cublasCreate(&cublasHandle));

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
    bool passing_start = false;
    double waiting_time = get_current_time(start);
    waiting_time = waiting_time + DYNAMIC_TERM;

    // bool first = true;
    while(input_index <= request_number){
        
        double input_time = get_current_time(start);

        passing_start = false;
        if(model.data_num == MAX_REQ_SIZE || input_time > waiting_time){
            if(model.data_num != MAX_REQ_SIZE) printf("!! current time : %lf, %lf\n", input_time, waiting_time);
            if(input_index != 0) passing_start = true;

            waiting_time = input_time + DYNAMIC_TERM;
        }

        /* New input condition */
        if(passing_start==false){
            if(input_index != request_number && input_time > request_vector.at(input_index)){
            
                struct vector_layer v_new = set_vector(0, input_index, request_unroll.at(input_index) );
                
                v_layer.push_back(v_new);
                if(v_layer.size()==1) waiting_time = input_time + DYNAMIC_TERM;

                input_index += 1;
                printf("--came at time : %lf--\n", input_time);
                printf("--input index : %d\n", input_index);
                printf("--waiting time : %lf\n", waiting_time);
            
                model.data_num += 1;
                if(merge_request(&v_layer))     printf("merged!\n");
            }
        }
        else{
            bool sig_end = false;
            if(input_index == request_number) sig_end = true;

            if(v_layer.size() <= 0) continue;
            // std::cout<<"current req num: "<<model.data_num<<", starts at "<<model.data_first<<std::endl;
            printf("size : %d\n", v_layer.size());
            assert(v_layer.size() == 1);
            

            int idx_end = v_layer.size()-1;
            struct vector_layer v_now = v_layer[idx_end];

            
            assert(v_now.idx_layer == 0);
            while(v_now.idx_layer != MAX_LAYER_NUM){
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
                    if(req_passed > 0) v_layer.push_back(v_now);
                    if(req_passed==0){
                        v_now.idx_layer += 1;
                        if(v_now.idx_layer >= MAX_LAYER_NUM){
                            int request_done = v_now.idx_request.size();
                            model.data_first += request_done;
                            model.data_num -= request_done;
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
            if(sig_end) break;
        }
    }

  for(int i=0; i<MAX_LAYER_NUM; i++){
      cudaFree(model.list_layer[i]->workSpace);
  }
  return 0;
}
