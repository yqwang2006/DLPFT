#ifndef ONEHOT_H
#define ONEHOT_H
#include "armadillo"
static arma::mat onehot(int rows, int cols, arma::mat labels){
	arma::mat desired_out = arma::zeros(rows,cols);
	
	for(int i = 0;i < cols; i++){
		if(labels(i) == 0) labels(i) = rows;
		desired_out(labels(i)-1,i) = 1;
		//desired_out(labels(i)-1,i) = 1;
	}

	return desired_out;
}
static arma::mat onehot_elm(int rows, int cols, arma::mat labels){
	arma::mat desired_out = -1 * arma::ones(rows,cols);
	
	for(int i = 0;i < cols; i++){
		if(labels(i) == 0) labels(i) = rows;
		desired_out(labels(i)-1,i) = 1;
		//desired_out(labels(i)-1,i) = 1;
	}

	return desired_out;
}
#endif