#ifndef __CROSSVALIDATIONKR_H
#define __CROSSVALIDATIONKR_H

#include <string>
#include <vector>

using namespace std;

void crossValidationKR(string EKIDX, int nfolds, int kernel, int dist_size, float* dev_dist, char* label_file, char* model_save_file, int gpu_id);


vector< vector<int> > split_folds(int* index, int n, int nfolds);


float calcap(float* labels, float* rank, int len);
#endif
