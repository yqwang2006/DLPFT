#ifndef CONVOLVE_H
#define CONVOLVE_H

#include "armadillo"

static arma::mat rot(arma::mat W,int k){
	if(k == 0) return W;
	arma::mat B = arma::zeros(W.n_rows,W.n_cols);
	if(k == 1){
		for(int i = 0;i < W.n_cols; i++){
			B.col(i) = W.col(W.n_cols-i-1);
		}
		B = B.t();
	}else if(k == 2){
		for(int i = 0;i < W.n_rows; i++){
			for(int j = 0;j < W.n_cols; j++){
				B(i,j) = W(W.n_rows-i-1,W.n_cols-j-1);
			}
		}
	}
	return B;
}

static arma::mat convn(const arma::mat image, const arma::mat W, string info){
	int image_dim,conv_dim;
	int filter_dim = W.n_cols;

	arma::mat feature;
	arma::mat filter;
	arma::mat full_image;
	if(info == "valid"){
		image_dim = image.n_cols;
		conv_dim = image_dim - filter_dim + 1;
		feature = arma::zeros(conv_dim,conv_dim);
		filter = W;
		full_image = image;
	}else if(info == "full"){
		image_dim = image.n_cols+2*(filter_dim-1);
		full_image = arma::zeros(image_dim,image_dim);
		full_image.submat(filter_dim-1,filter_dim-1,filter_dim+image.n_rows-2,filter_dim+image.n_cols-2) = image;
		
		
		conv_dim = image_dim - filter_dim + 1;
		feature = arma::zeros(conv_dim,conv_dim);
		filter = rot(W,2);
	}

	for(int i = 0;i < conv_dim; i++){
		for(int j = 0;j < conv_dim;j++){
			arma::mat patch = full_image.submat(i,j,i+filter_dim-1,j+filter_dim-1);
			feature(i,j) = arma::sum(arma::sum(patch % filter));
		}
	}


	return feature;
}


//TODO: 实现convn的full和conv2
#endif