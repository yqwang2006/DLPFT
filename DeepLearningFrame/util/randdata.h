#ifndef RANDDATA_H
#define RANDDATA_H
#include "armadillo"
#include <vector>
using namespace std;
static void rand_data(const arma::mat input, arma::mat* batches,int sample_num, int batch_size){
	
	srand(unsigned(time(NULL)));
	int batches_num = sample_num / batch_size;
	int visible_size = input.n_rows;
	vector<int> groups;
	for(int i = 0;i < batch_size; i++){
		for(int j = 0;j < batches_num; j++){
			groups.push_back(j);
		}
	}

	random_shuffle(groups.begin(),groups.end());

	arma::mat groups_mat = arma::zeros(groups.size(),1);
	for(int i = 0;i < groups.size(); i++)
	{
		groups_mat(i) = groups[i];
	}

	for(int i = 0;i < batches_num; i++){
		batches[i] = input.cols(find(groups_mat == i));
	}


}

#endif