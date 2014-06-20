#ifndef ONEHOT_H
#define ONEHOT_H
#include "armadillo"
static arma::mat onehot(int rows, int cols, arma::imat labels){
	arma::mat desired_out = arma::zeros(rows,cols);
	for(int i = 0;i < cols; i++){
		
			desired_out(labels(i)-1,i) = 1;
		//desired_out(labels(i)-1,i) = 1;
	}

	return desired_out;
}

#endif