#include <iostream>
#include <fstream>
#include <assert.h>
#include "KR.h"
#include "crossValidationKR.h"
using namespace std;

int main(int argc, char* argv[])
{
	if (argc != 9)
	{
		cout << "Usage: ./CrossValidation EKIDX nfolds kernel dist_size dist_file label_file model_save_file gpu_id" << endl;
		return 1;
	}

	string EKIDX = argv[1];
	int nfolds = atoi(argv[2]);
	int kernel = atoi(argv[3]);//6
	int dist_size = atoi(argv[4]);//8030
	int gpu_id = atoi(argv[8]);

	char* dist_name = argv[5];
	//char* dist_name = "dist";

	cout << "start reading distance feature" << endl;

	size_t pos = 0;

	int ridx;
	float rval;
	float* dev_dist = (float*)malloc(sizeof(float)*(dist_size*dist_size));//Load data from 
	assert(dev_dist != NULL);
	FILE* dist_file = fopen(dist_name, "r");

	while(fscanf(dist_file, "%d:%f", &ridx, &rval)!=EOF)
	{
		if(ridx != 0)
		{
			dev_dist[pos] = rval;
			pos ++;
		}
	}
	fclose(dist_file);
	assert(pos == dist_size * dist_size);

	cout << "Finish reading distance feature" << endl;
	char* label_file = argv[6];
	char* model_save_file = argv[7];
	crossValidationKR(EKIDX, nfolds, kernel, dist_size, dev_dist, label_file, model_save_file, gpu_id);

	free(dev_dist);

	return 0;
}

