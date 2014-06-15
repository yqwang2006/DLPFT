#ifndef ONEHOT_H
#define ONEHOT_H
#include "armadillo"
static arma::mat onehot(int rows, int cols, arma::imat labels){
	arma::mat desired_out = arma::zeros(rows,cols);
	for(int i = 0;i < cols; i++){
		if(labels(i) == rows)
			desired_out(0,i) = 1;
		else
			desired_out(labels(i),i) = 1;
	}

	return desired_out;
}

#endif