#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include "KR.h"

#include <assert.h>

using namespace std;

KR::KR(float* KerTrain, float* KerTrainTest, float* Y_train, float _lambda, int training_num_, int testing_num_, int gpu_id_)
{
    assert(KerTrain != NULL);
    assert(KerTrainTest != NULL);
    assert(Y_train != NULL);
    training_num = training_num_;
    testing_num = testing_num_;
    KerTr = (float*)malloc(sizeof(float)*training_num*training_num);
    assert(KerTr != NULL);
    for(int i_tr = 0; i_tr < training_num*training_num; i_tr++)
    {
        *(KerTr + i_tr) = *(KerTrain+i_tr);
    }
    KerTrTe = (float*)malloc(sizeof(float)*training_num*testing_num);
    assert(KerTrTe != NULL);
    for(int i_te = 0; i_te < training_num * testing_num; i_te++)
    {
        *(KerTrTe + i_te) = *(KerTrainTest + i_te);
    }


    Y_tr = (float*)malloc(sizeof(float)*training_num);
    assert(Y_tr != NULL);
    for(int i_tr = 0; i_tr < training_num; i_tr++)
    {
        *(Y_tr + i_tr) = *(Y_train + i_tr);
    }
    lambda = _lambda;

    gpu_id = gpu_id_;

    Y_Pre = (float*)malloc(sizeof(float)*testing_num);
    assert(Y_Pre != NULL);
    bias = (float*)malloc(sizeof(float));
    assert(bias != NULL);

    W = (float*)malloc(sizeof(float)*training_num);
    assert(W != NULL);
}

KR::KR(float* KerTrain, float* Y_train, float _lambda, int training_num_, int gpu_id_)
{
    assert(KerTrain != NULL);
    assert(Y_train != NULL);
    lambda = _lambda;
    training_num = training_num_;
    KerTr = (float*)malloc(sizeof(float)*training_num*training_num);
    assert(KerTr != NULL);
    for(int i_tr = 0; i_tr < training_num*training_num; i_tr++)
    {
        *(KerTr + i_tr) = *(KerTrain + i_tr);
    }
    Y_tr = (float*)malloc(sizeof(float)*training_num);
    assert(Y_tr);
    for(int i_tr = 0; i_tr < training_num; i_tr++)
    {
        *(Y_tr + i_tr) = *(Y_train + i_tr);
    }

    gpu_id = gpu_id_;

    bias = (float*)malloc(sizeof(float));
    assert(bias != NULL);
    W = (float*)malloc(sizeof(float)*training_num);
    assert(W != NULL);
}

KR::~KR()
{

    free(KerTr);
    if(KerTrTe != NULL)
    {
        free(KerTrTe);
    }
    free(Y_tr);
    free(Y_Pre);
    free(W);
}

void KR::checkStatus(culaStatus status)
{
    char buf[256];
    if(!status)
        return;
    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    cout << "CULA Exception: " << buf << endl;

    culaShutdown();
    exit(EXIT_FAILURE);
}

void KR::print_matrix(const float* A, int nr_rows_A, int nr_cols_A)
{
    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            cout << A[j* nr_rows_A + i] << " ";
        }
        cout << endl;
    }
/*
    for(int i = 0; i < nr_rows_A * nr_cols_A; i++)
        cout << A[i] << " ";
    cout << endl;
    */
}

void KR::compute()
{
    culaSelectDevice(gpu_id);
    culaStatus status;
    status = culaInitialize();
    checkStatus(status);

    float* h_I = NULL;
    float* h_ones = NULL;

    h_I = (float*)malloc(sizeof(float) * training_num * training_num);
    assert( h_I != NULL);
    h_ones = (float*)malloc(sizeof(float) * training_num);
    assert( h_ones != NULL);

    size_t j = 0;
    for(size_t i = 0; i < training_num * training_num; i++)
    {
        if(i % training_num == j)
        {
            h_I[i] = 1;
        }
        else
            h_I[i] = 0;
        if((i+1) % training_num == 0)
            j++;
    }
    for(j = 0; j < training_num; j++)
        h_ones[j] = 1;

    float* d_KerTr = NULL;
    int* ipiv = NULL;
    float* d_I = NULL;
    float* d_H = NULL;
    float* d_ones = NULL;
    float* d_HKernel = NULL;
    float* d_YTrain = NULL;
    float* d_Yinvk = NULL;
    float* d_Yinvk_K = NULL;
    float* d_bias;
    float* d_YTrinvK = NULL;
    float* d_YPre = NULL;
    float* d_KerTrTe = NULL;

    cudaMalloc(&d_KerTr, sizeof(float)*training_num * training_num);
    cudaMalloc(&d_YTrain, sizeof(float)*training_num);
    cudaMalloc(&d_I, sizeof(float)*training_num*training_num);
    
    cudaMalloc(&d_H, sizeof(float)*training_num*training_num);
    cudaMalloc(&d_HKernel, sizeof(float)* training_num * training_num);
    cudaMalloc(&d_ones, sizeof(float)* training_num);
    cudaMalloc(&ipiv, sizeof(int)*training_num);

    cudaMalloc(&d_Yinvk, sizeof(float)*training_num);
    cudaMalloc(&d_Yinvk_K, sizeof(float)*training_num);
    cudaMalloc(&d_KerTrTe, sizeof(float)*training_num*testing_num);
    cudaMalloc(&d_bias, sizeof(float));

    cudaMalloc(&d_YPre, sizeof(float)*testing_num);
    cudaMalloc(&d_YTrinvK, sizeof(float)*training_num);

    cudaMemcpy(d_I, h_I, sizeof(float)* training_num*training_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, h_I, sizeof(float)* training_num * training_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ones, h_ones, sizeof(float)* training_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_YTrain, Y_tr, sizeof(float)*training_num, cudaMemcpyHostToDevice);

    cudaMemcpy(d_KerTr, KerTr,  sizeof(float)*training_num * training_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_KerTrTe, KerTrTe, sizeof(float)*training_num*testing_num, cudaMemcpyHostToDevice);

    status = culaDeviceSgemm('N', 'N', training_num, training_num, 1, -1.0/training_num, d_ones, training_num, d_ones, 1, 1.0, d_H, training_num);
    checkStatus(status);
    
    status = culaDeviceSgemm('N', 'N', training_num, training_num, training_num, 1, d_H, training_num, d_KerTr, training_num, 0.0, d_HKernel, training_num);
    checkStatus(status);

    status = culaDeviceSgemm('N', 'N', training_num, training_num, training_num, 1, d_HKernel, training_num, d_H, training_num, lambda, d_I, training_num);
    checkStatus(status);

    // invK: invI*H
    status = culaDeviceSgesv(training_num, training_num, d_I, training_num, ipiv, d_H, training_num);
    checkStatus(status);
    // d_H is invK
    //
    

    // bias
    status = culaDeviceSgemm('T', 'N', 1, training_num, training_num, 1, d_YTrain, training_num, d_H, training_num, 0, d_Yinvk, 1);
    checkStatus(status);
    status = culaDeviceSgemm('N', 'T', 1, training_num, training_num, 1, d_Yinvk, 1, d_KerTr, training_num, 0, d_Yinvk_K, 1);
    checkStatus(status);
    status = culaDeviceSgemm('N', 'N', 1, 1, training_num, 1, d_Yinvk_K, 1, d_ones, training_num, 0, d_bias, 1);
    checkStatus(status);

    status = culaDeviceSgemm('N', 'N', 1, 1, training_num, 1.0/training_num, d_YTrain, 1, d_ones, training_num, -1.0/training_num, d_bias, 1);
    checkStatus(status);
    // bias end

    cudaMemcpy(W, d_Yinvk, sizeof(float)*training_num, cudaMemcpyDeviceToHost);

    status = culaDeviceSgemm('N', 'T', 1, testing_num, training_num, 1, d_Yinvk, 1, d_KerTrTe, testing_num, 0, d_YPre, 1);
    checkStatus(status);
    
//  status = culaDeviceSgemm('N', 'N', 1, testing_num, training_num, 1, d_YTrinvK, 1, d_KerTrTe, training_num, 0, d_YPre, 1);
//    checkStatus(status);

    cudaMemcpy(Y_Pre, d_YPre, sizeof(float)*testing_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);


    for(int i = 0; i < testing_num; i++)
    {
        *(Y_Pre + i) = *(Y_Pre + i) + *bias;
    }

    culaShutdown();

    free(h_I);
    free(h_ones);

    cudaFree(d_KerTr);
    cudaFree(d_YTrain);
    cudaFree(d_H);
    cudaFree(d_I);
    cudaFree(d_ones);
    cudaFree(d_KerTr);
    cudaFree(d_KerTrTe);
}

void KR::computeW()
{
    culaSelectDevice(gpu_id);
    culaStatus status;
    status = culaInitialize();
    checkStatus(status);

    float* h_I = NULL;
    float* h_ones = NULL;

    h_I = (float*)malloc(sizeof(float) * training_num * training_num);
    assert( h_I != NULL);
    h_ones = (float*)malloc(sizeof(float) * training_num);
    assert( h_ones != NULL);

    size_t j = 0;
    for(size_t i = 0; i < training_num * training_num; i++)
    {
        if(i % training_num == j)
        {
            h_I[i] = 1;
        }
        else
            h_I[i] = 0;
        if((i+1) % training_num == 0)
            j++;
    }
    for(j = 0; j < training_num; j++)
        h_ones[j] = 1;

    float* d_KerTr = NULL;
    int* ipiv = NULL;
    float* d_I = NULL;
    float* d_H = NULL;
    float* d_ones = NULL;
    float* d_HKernel = NULL;
    float* d_YTrain = NULL;
    float* d_Yinvk = NULL;
    float* d_Yinvk_K = NULL;
    float* d_bias;
    float* d_YTrinvK = NULL;

    cudaMalloc(&d_KerTr, sizeof(float)*training_num * training_num);
    cudaMalloc(&d_YTrain, sizeof(float)*training_num);
    cudaMalloc(&d_I, sizeof(float)*training_num*training_num);
    
    cudaMalloc(&d_H, sizeof(float)*training_num*training_num);
    cudaMalloc(&d_HKernel, sizeof(float)* training_num * training_num);
    cudaMalloc(&d_ones, sizeof(float)* training_num);
    cudaMalloc(&ipiv, sizeof(int)*training_num);

    cudaMalloc(&d_Yinvk, sizeof(float)*training_num);
    cudaMalloc(&d_Yinvk_K, sizeof(float)*training_num);
    cudaMalloc(&d_bias, sizeof(float));

    cudaMalloc(&d_YTrinvK, sizeof(float)*training_num);

    cudaMemcpy(d_I, h_I, sizeof(float)* training_num*training_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, h_I, sizeof(float)* training_num * training_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ones, h_ones, sizeof(float)* training_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_YTrain, Y_tr, sizeof(float)*training_num, cudaMemcpyHostToDevice);

    cudaMemcpy(d_KerTr, KerTr,  sizeof(float)*training_num * training_num, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    status = culaDeviceSgemm('N', 'N', training_num, training_num, 1, -1.0/training_num, d_ones, training_num, d_ones, 1, 1.0, d_H, training_num);
    checkStatus(status);
/*
    float* hhh_H = (float*)malloc(sizeof(float)*training_num*training_num);
    cudaMemcpy(hhh_H, d_H, sizeof(float)*training_num*training_num, cudaMemcpyDeviceToHost);

    print_matrix(hhh_H, training_num, training_num);
    free(hhh_H);
    cout << endl;
*/    
    status = culaDeviceSgemm('N', 'N', training_num, training_num, training_num, 1, d_H, training_num, d_KerTr, training_num, 0.0, d_HKernel, training_num);
    checkStatus(status);
/*
    float* hhh_HKernel = (float*)malloc(sizeof(float)*training_num*training_num);
    cudaMemcpy(hhh_HKernel, d_HKernel, sizeof(float)*training_num*training_num, cudaMemcpyDeviceToHost);
    print_matrix(hhh_HKernel, training_num, training_num);
    free(hhh_HKernel);
    cout << endl;
*/

    status = culaDeviceSgemm('N', 'N', training_num, training_num, training_num, 1, d_HKernel, training_num, d_H, training_num, lambda, d_I, training_num);
    checkStatus(status);

/*    
    hhh_HKernel = (float*)malloc(sizeof(float)*training_num*training_num);
    cudaMemcpy(hhh_HKernel, d_I, sizeof(float)*training_num*training_num, cudaMemcpyDeviceToHost);
    print_matrix(hhh_HKernel, training_num, training_num);
    free(hhh_HKernel);
    cout << endl;
*/

    // invK: invI*H
    status = culaDeviceSgesv(training_num, training_num, d_I, training_num, ipiv, d_H, training_num);
    checkStatus(status);
    // d_H is invK
    //
/*   
    float* hhhh_H = (float*)malloc(sizeof(float)*training_num*training_num);
    cudaMemcpy(hhhh_H, d_H, sizeof(float)*training_num*training_num, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    print_matrix(hhhh_H, training_num, training_num);
    free(hhhh_H);
    cout << endl;
*/    


    // bias
    status = culaDeviceSgemm('T', 'N', 1, training_num, training_num, 1, d_YTrain, training_num, d_H, training_num, 0, d_Yinvk, 1);
    checkStatus(status);
    status = culaDeviceSgemm('N', 'T', 1, training_num, training_num, 1, d_Yinvk, 1, d_KerTr, training_num, 0, d_Yinvk_K, 1);
    checkStatus(status);
    status = culaDeviceSgemm('N', 'N', 1, 1, training_num, 1, d_Yinvk_K, 1, d_ones, training_num, 0, d_bias, 1);
    checkStatus(status);

    status = culaDeviceSgemm('N', 'N', 1, 1, training_num, 1.0/training_num, d_YTrain, 1, d_ones, training_num, -1.0/training_num, d_bias, 1);
    checkStatus(status);
    // bias end

    cudaMemcpy(W, d_Yinvk, sizeof(float)*training_num, cudaMemcpyDeviceToHost);
/*
    for(int i = 0; i < 5; i++)
        cout << W[i] << " ";
    cout << endl;
*/
    cudaMemcpy(bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);

//    cout << *bias << endl;

    cudaDeviceSynchronize();



    culaShutdown();

    free(h_I);
    free(h_ones);

    cudaFree(d_KerTr);
    cudaFree(d_YTrain);
    cudaFree(d_H);
    cudaFree(d_I);
    cudaFree(d_ones);
    cudaFree(d_KerTr);
}



int KR::MeetsMinimumCulaRequirements()
{
    int cudaMinimumVersion = culaGetCudaMinimumVersion();
    int cudaRuntimeVersion = culaGetCudaRuntimeVersion();
    int cudaDriverVersion = culaGetCudaDriverVersion();
    int cublasMinimumVersion = culaGetCublasMinimumVersion();
    int cublasRuntimeVersion = culaGetCublasRuntimeVersion();
    if(cudaRuntimeVersion < cudaMinimumVersion)
    {
        printf("CUDA runtime version is insufficient; "
                "version %d or greater is required\n", cudaMinimumVersion);
        return 0;
    }
    if(cudaDriverVersion < cudaMinimumVersion)
    {
        printf("CUDA driver version is insufficient; "
                "version %d or greater is required\n", cudaMinimumVersion);
        return 0;
    }
    if(cublasRuntimeVersion < cublasMinimumVersion)
    {
        printf("CUBLAS runtime version is insufficient; "
                "version %d or greater is required\n", cublasMinimumVersion);
        return 0;
    }
    return 1;
}



float* KR::get_YPre()
{
    return Y_Pre;
}
float* KR::get_W()
{
    return W;
}
float* KR::get_bias()
{
    return bias;
}
class KR;
