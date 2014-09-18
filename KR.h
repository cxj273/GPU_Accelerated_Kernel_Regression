#ifndef __KR_H
#define __KR_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cula.h>
#include <cula_lapack.h>
#include <cula_blas.h>
#include <cula_blas_device.h>

// Kernel Regression
//
//

class KR
{
public:
    KR(float* KerTr, float* KerTrTe, float* Y_tr, float lambda, int training_num, int testing_num, int gpu_id_);
    KR(float* KerTrain, float* Y_train, float _lambda, int training_num_, int gpu_id_);
    ~KR( );

    void computeW();
    void compute();
    void checkStatus(culaStatus status);
    float* get_YPre();
    float* get_W();
    float* get_bias();
    void print_matrix(const float* A, int nr_rows_A, int nr_cols_A);
    int MeetsMinimumCulaRequirements();

private:
    float* KerTr;
    float* KerTrTe;
    float* Y_tr;
    float* W;
    float lambda;
    int training_num;
    int testing_num;
    int gpu_id;
    float* Y_Pre;
    float* bias;
};
#endif
