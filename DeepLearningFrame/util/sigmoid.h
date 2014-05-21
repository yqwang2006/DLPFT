#ifndef SIGMOID_H
#define SIGMOID_H
#include "armadillo"
#include <vector>
static arma::mat sigmoid(const arma::mat x){
	//clock_t start,end;
	//double dur;
	//
	//start = clock();

	
	arma::mat s(x.n_rows,x.n_cols);
	s = 1/(1+arma::exp(-x));

	//end = clock();
	//dur = (double)(end-start)/CLOCKS_PER_SEC;
	//cout << "sig:" << dur << "s" << endl;
	return s;
}

static arma::mat sigmoidInv(arma::mat& x){
	//clock_t start,end;
	//double dur;
	//
	//start = clock();
	arma::mat s(x.n_rows,x.n_cols);
	s = sigmoid(x) % (1-sigmoid(x));
	//end = clock();
	//dur = (double)(end-start)/CLOCKS_PER_SEC;
	//cout << "sigInv:" << dur << "s" << endl;
	return s;
}

static std::vector<int>& randperm(int N)
{
	std::vector<int> rand_perm;
	for(int i =0;i < N;i++){
		rand_perm.push_back(i);
	}
	random_shuffle(rand_perm.begin(),rand_perm.end());
	
	return rand_perm;
}
#endif