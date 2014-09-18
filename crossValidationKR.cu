
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <cula.h>
#include <cula_lapack.h>
#include <cula_blas.h>
#include <cula_blas_device.h>

#include <assert.h>

#include "crossValidationKR.h"
#include "KR.h"

#define MINIMUM (0.000000001)

using namespace std;

void crossValidationKR(string EKIDX, int nfolds, int kernel, int dist_size, float* dev_dist, char* label_file, char* model_save_file, int gpu_id)
{
    string line;
    ifstream ifs;
    ifs.open(label_file);
    int n_pos = 0;
    int n_neg = 0;
    while(getline(ifs, line))
    {
        float val = (float)atof(line.c_str());
        if(val < MINIMUM && val > - MINIMUM)
        {
            n_neg ++;
        }
        if(val > MINIMUM && (val - 1.0) < MINIMUM)
        {
            n_pos ++;
        }
    }
    ifs.close();
    int num_used = n_neg + n_pos;
    cout << "n_neg: " << n_neg << endl;
    cout << "n_pos: " << n_pos << endl;

    ifs.open(label_file);
    int* index_valid = (int*)malloc(sizeof(int) * num_used);
    assert(index_valid != NULL);
    float* label_list = (float*)malloc(sizeof(float) * num_used);
    assert(label_list != NULL);
    int index = 0, acc = 0;
    while(getline(ifs, line))
    {
        float val = (float)atof(line.c_str());
        if(val > - MINIMUM)
        {
            label_list[acc] = val;
            index_valid[acc] = index;
            acc++;
        }
        index ++;
    }
    assert(acc == num_used);
    assert(index == dist_size);
    ifs.close();
    float* dev_dist_used = (float*)malloc(sizeof(float) * num_used * num_used);
    assert(dev_dist_used != NULL);
    for(int i = 0; i < num_used; i++)
        for(int j = 0; j < num_used; j++)
        {
            *(dev_dist_used + i * num_used + j) = *(dev_dist + index_valid[i] * dist_size + index_valid[j]);
        }

    int* index_pos = (int*)malloc(sizeof(int) * n_pos);
    assert(index_pos != NULL);
    int* index_neg = (int*)malloc(sizeof(int) * n_neg);
    assert(index_neg != NULL);
    int idx_pos = 0;
    int idx_neg = 0;
    for(int i = 0; i < num_used; i++)
    {
        if(label_list[i] < MINIMUM && label_list[i] > -MINIMUM)
        {
            index_neg[idx_neg] = i;
            idx_neg++;
        }
        if(label_list[i] > MINIMUM && (label_list[i] - 1.0) < MINIMUM)
        {
            index_pos[idx_pos] = i;
            idx_pos++;
        }
    }

    vector< vector<int> > pos_folds = split_folds(index_pos, n_pos, nfolds);
    vector< vector<int> > neg_folds = split_folds(index_neg, n_neg, nfolds);

    float best_ap = -1;
    float best_lambda = -1;
    float best_gamma = -1;
    float threshold = 0;

    vector<float> gamma_range;
    if(strcmp(EKIDX.c_str(), "EK100") == 0 && kernel == 7)
    {
        /*
           gamma_range.push_back(pow(2, -15));
           gamma_range.push_back(pow(2, -13));
           gamma_range.push_back(pow(2, -11));
           gamma_range.push_back(pow(2, -9));
           gamma_range.push_back(pow(2, -7));
           gamma_range.push_back(pow(2, -5));
           gamma_range.push_back(pow(2, -3));
           gamma_range.push_back(pow(2, -1));
           gamma_range.push_back(2);
           gamma_range.push_back(pow(2, 3));
         */
    }
    else
    {
        gamma_range.push_back(1);
    }

    float* dev_kernel = (float*)malloc(sizeof(float) * num_used * num_used);
    assert(dev_kernel != NULL);
    for(int i_gamma = 0; i_gamma < gamma_range.size(); i_gamma++)
    {
        float gamma = gamma_range[i_gamma];
        if(kernel == 6)
        {
            if(strcmp(EKIDX.c_str(), "EK10") == 0)
            {
                float sum = 0.0;
                for(int i = 0; i < num_used * num_used; i++)
                {
                    sum = sum + fabs(dev_dist_used[i]);
                }
                sum = sum / (num_used * num_used);
                for(int i = 0; i < num_used * num_used; i++)
                {
                    dev_dist_used[i] = dev_dist_used[i] / sum;
                }
            }


            for(int i = 0; i < num_used * num_used; i++)
            {
                dev_kernel[i] = dev_dist_used[i];
            }
        }
        else
        {
            for(int i = 0; i < num_used * num_used; i++)
            {
                dev_kernel[i] = std::exp(-gamma * dev_dist_used[i]);
            }
        }
        vector<float> lambda_range;
        if(strcmp(EKIDX.c_str(), "EK100") == 0)
        {
            lambda_range.push_back(0.0001);
            lambda_range.push_back(0.01);
            lambda_range.push_back(1);
            lambda_range.push_back(100);
            lambda_range.push_back(10000);
        }
        else
        {
            lambda_range.push_back(1);
        }

        vector<float> best_cv_list;
        vector<int> best_cv_idx;
        for(int i_lambda = 0; i_lambda < lambda_range.size(); i_lambda++)
        {
            float lambda = lambda_range[i_lambda];
            float acc_ap = 0;

            cout << "lambda = " << lambda << endl;

            vector<float> learn_threshold_predict;
            vector<int> all_idx;

            for(int u = 0; u < nfolds; u++)
            {
                vector<int> training_idx;
                vector<int> testing_idx;

                for(int v = 0; v < nfolds; v++)
                {
                    if(u == v)
                    {
                        testing_idx.insert(testing_idx.end(), pos_folds[v].begin(), pos_folds[v].end());
                        testing_idx.insert(testing_idx.end(), neg_folds[v].begin(), neg_folds[v].end());
                    }
                    else
                    {
                        training_idx.insert(training_idx.end(), pos_folds[v].begin(), pos_folds[v].end());
                        training_idx.insert(training_idx.end(), neg_folds[v].begin(), neg_folds[v].end());
                    }
                }

                std::sort(testing_idx.begin(), testing_idx.end());
                std::sort(training_idx.begin(), training_idx.end());
                int training_num = training_idx.size();
                int testing_num = testing_idx.size();
                cout << "training_num: " << training_num << endl;
                cout << "testing_num: " << testing_num << endl;
                float* train_dist = (float*)malloc(sizeof(float) * training_num * training_num);
                assert(train_dist != NULL);
                for(int i_train = 0; i_train < training_num; i_train++)
                    for(int j_train = 0; j_train < training_num; j_train++)
                    {
                        *(train_dist + i_train * training_num + j_train) = *(dev_kernel + training_idx[i_train] * num_used + training_idx[j_train]);
                    }
                float* test_dist = (float*)malloc(sizeof(float) * training_num * testing_num);
                assert(test_dist != NULL);
                for(int i_train = 0; i_train < training_num; i_train++)
                    for(int i_test = 0; i_test < testing_num; i_test++)
                    {
                        *(test_dist + i_train * testing_num + i_test) = *(dev_kernel + training_idx[i_train] * num_used + testing_idx[i_test]);
                    }

                float* training_labels = (float*)malloc(sizeof(float) * training_num);
                assert(training_labels != NULL);
                for(int i_train = 0; i_train < training_num; i_train++)
                {
                    *(training_labels + i_train) = label_list[training_idx[i_train]];
                }

                float* testing_labels = (float*)malloc(sizeof(float) * testing_num);
                for(int i_test = 0; i_test < testing_num; i_test++)
                {
                    *(testing_labels + i_test) = label_list[testing_idx[i_test]];
                }

                float* Y_pre = NULL;
                Y_pre = (float*)malloc(sizeof(float) * testing_num);
                assert(Y_pre != NULL);

                KR* KerRegre = new KR(train_dist, test_dist, training_labels, lambda, training_num, testing_num, gpu_id);

                KerRegre->compute();
                Y_pre = KerRegre->get_YPre();
                //KerRegre->print_matrix(Y_pre, 1, 2);

                for(int kk = 0; kk < testing_num; kk++)
                {
                    all_idx.push_back(index_valid[testing_idx[kk]]);
                    learn_threshold_predict.push_back(Y_pre[kk]);
                    //fprintf(stderr, "%d %d %d %g\n", kk, testing_idx[kk], index_valid[testing_idx[kk]], Y_pre[kk]);
                }


                float ap;
                ap = calcap(testing_labels, Y_pre, testing_num);

                cout << "ap = " << ap << endl;

                acc_ap = acc_ap + ap;

                free(train_dist);
                free(test_dist);
                free(training_labels);
                free(testing_labels);
                free(Y_pre);
            }

            acc_ap = acc_ap / nfolds;

            if(acc_ap > best_ap)
            {
                best_ap = acc_ap;
                best_lambda = lambda;
                best_gamma = gamma;
                best_cv_list = learn_threshold_predict;
                best_cv_idx = all_idx;
            }

        }


        cout << "best ap: " << best_ap << endl;
        cout << "best lambda :" << best_lambda << endl;

        float* W = (float*)malloc(sizeof(float) * num_used);
        assert(W != NULL);
        float* W_bias = (float*)malloc(sizeof(float));
        assert(W_bias != NULL);
        KR* KerRegre = new KR(dev_kernel, label_list, best_lambda, num_used, gpu_id);
        KerRegre->computeW();
        W = KerRegre->get_W();
        W_bias = KerRegre->get_bias();

        ofstream ofs;
        ofs.open(model_save_file);
        ofs << "label 1 0" << endl;
        ofs << "avg 1" << endl;
        ofs << "rho " << (*W_bias)*(-1) << endl;
        ofs << "total_sv " << num_used << endl;
        ofs << "SV" << endl;
        for(int i = 0; i < num_used; i++)
        {
            ofs << W[i] << " 0:" << index_valid[i] + 1 << endl;
        };
        ofs.close();

        string train_predict = string(model_save_file) + ".predict";
        FILE* out = fopen(train_predict.c_str(), "w");
        int l = best_cv_idx.size();
        for(int kk = 0; kk < l; kk++)
        {
            fprintf(out, "%d %g\n", best_cv_idx[kk] + 1, best_cv_list[kk]);
        }
        fclose(out);

        free(W);
        free(W_bias);
    }

}

vector< vector<int> > split_folds(int* index, int n, int nfolds)
{
    vector< vector<int> > folds;
    int left = n % nfolds;
    int div = (n - left) / nfolds;
    int acc = 0;
    for(int u = 1; u <= nfolds; u++)
    {
        int max_idx = div;
        if(u <= left)
        {
            max_idx = div + 1;
        }
        vector<int> this_folder;
        for(int v = 0; v < max_idx; v++)
        {
            this_folder.push_back(index[acc]);
            acc = acc + 1;
        }
        folds.push_back(this_folder);
    }

    if(acc != n)
    {
        std::cout << "Split folds failed! " << acc << " != " << n << std::endl;
        exit(1);
    }
    return folds;
}


float calcap(float* labels, float* rank, int len)
{

    int* idx = (int*) malloc(len * sizeof(int));
    assert(idx != NULL);

    for(int i = 0; i < len; ++i)
    {
        idx[i] = i;
    }

    for(int i = 0; i < len; ++i)
    {
        for(int j = i + 1; j < len; j++)
        {
            if(rank[i] < rank[j])
            {
                //swap
                float temp = rank[i];
                rank[i] = rank[j];
                rank[j] = temp;

                //swap idx
                int temp_idx = idx[i];
                idx[i] = idx[j];
                idx[j] = temp_idx;
            }
        }
    }

    int poss = 0;
    for(int i = 0; i < len; i++)
    {
        if(labels[i] == 1)
        {
            poss ++;
        }
    }

    float accpos = 0;
    float ap = 0;

    for(int u = 0; u < len; u++)
    {
        if(labels[idx[u]] == 1)
        {
            accpos = accpos + 1;
        }

        if(u == (len - 1))
        {
        }
        else if(rank[u] != rank[u + 1])
        {
        }

        else
        {
            continue;
        }

        if(accpos != 0)
        {
            vector<int> above_threshold;
            for(int j = 0; j <=  u; j++)
            {
                above_threshold.push_back(idx[j]);
            }
            vector<float> retrieved;
            for(int j = 0; j <= u; j++)
            {
                retrieved.push_back(labels[above_threshold[j]]);
            }
            int re_pos = 0;
            for(int t = 0; t <= u; t++)
            {
                if(retrieved[t] == 1)
                {
                    re_pos ++;
                }
            }

            ap = ap + (float) re_pos / (u + 1) * accpos;
            accpos = 0;
        }
    }
    ap = ap / poss;

    free(idx);

    return ap;
}



