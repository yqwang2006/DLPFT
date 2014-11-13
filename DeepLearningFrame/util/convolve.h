#ifndef CONVOLVE_H
#define CONVOLVE_H

#include "armadillo"


static arma::mat rot(arma::mat W,int k){
	if(k == 0) return W;
	arma::mat B = arma::zeros(W.n_rows,W.n_cols);
	if(k == 1){
#ifdef OPENMP
#pragma omp parallel for shared(B,W)
#endif
		for(int i = 0;i < W.n_cols; i++){
			B.col(i) = W.col(W.n_cols-i-1);
		}
		B = B.t();
	}else if(k == 2){
#ifdef OPENMP
#pragma omp parallel for shared(B,W)
#endif
		for(int i = 0;i < W.n_rows; i++){
			for(int j = 0;j < W.n_cols; j++){
				B(i,j) = W(W.n_rows-i-1,W.n_cols-j-1);
			}
		}
	}
	return B;
}
static arma::cube rot(arma::cube W,int k){
	if(k == 0) return W;
	arma::cube B;
	if(k == 1)
		B = arma::zeros(W.n_cols,W.n_rows,W.n_slices);
	else if(k == 2)
		B = arma::zeros(W.n_rows,W.n_cols,W.n_slices);
	for(int i = 0;i < W.n_slices;i++){
		
		if(k == 1){
			for(int c = 0;c < W.n_cols; c++){
				B.slice(i).col(c) = W.slice(i).col(W.n_cols-c-1);
				B.slice(i) = B.slice(i).t();
			}
			
		}else if(k == 2){
			for(int r = 0;r < W.n_rows; r++){
				for(int c = 0;c < W.n_cols;c++){
					B(i,r,c) = W(i,W.n_rows-r-1,W.n_cols-c-1);
				}
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
static arma::cube convn_cube(const arma::cube images, const arma::mat W, string info){
	int image_dim,conv_dim;
	int filter_dim = W.n_cols;
	int sample_num = images.n_slices;
	arma::cube feature,patch,feature_map,full_image;
	arma::mat filter;
	if(info == "valid"){
		image_dim = images.n_rows;
		conv_dim = image_dim - filter_dim + 1;
		feature = arma::zeros(conv_dim,conv_dim,sample_num);
		filter = W;
		full_image = images;
	}else{
		image_dim = images.n_cols+2*(filter_dim-1);
		full_image = arma::zeros(image_dim,image_dim,sample_num);
		full_image.tube(filter_dim-1,filter_dim-1,filter_dim+images.n_rows-2,filter_dim+images.n_cols-2) = images;
		
		
		conv_dim = image_dim - filter_dim + 1;
		feature = arma::zeros(conv_dim,conv_dim,sample_num);
		filter = rot(W,2);
	}
	arma::cube filter_scale = zeros(filter_dim,filter_dim,sample_num);	

	for(int i = 0;i < sample_num; i++){
		filter_scale.slice(i) = filter;
	}
#ifdef OPENMP
#pragma omp parallel for shared(conv_dim,full_image,filter_scale,filter_dim,feature)
#endif
	for(int i = 0;i < conv_dim; i++){
		for(int j = 0;j < conv_dim;j++){
			patch = full_image.tube(i,j,i+filter_dim-1,j+filter_dim-1);
			feature_map = patch % filter_scale;
			for(int k = 0;k < sample_num; k++){
				
				feature(i,j,k) = arma::sum(arma::sum(feature_map.slice(k)));
			}
		}
	}
	

	return feature;
}
static arma::cube convn_cube(const arma::cube images, const arma::cube W, string info){
	int image_dim,conv_dim;
	int filter_dim = W.n_cols;
	int sample_num = images.n_slices;
	arma::cube feature,patch,feature_map,filter,full_image;
	if(info == "valid"){
		image_dim = images.n_rows;
		conv_dim = image_dim - filter_dim + 1;
		feature = arma::zeros(conv_dim,conv_dim,sample_num);
		filter = W;
		full_image = images;
	}else{
		image_dim = images.n_cols+2*(filter_dim-1);
		full_image = arma::zeros(image_dim,image_dim,sample_num);
		full_image.tube(filter_dim-1,filter_dim-1,filter_dim+images.n_rows-2,filter_dim+images.n_cols-2) = images;
		
		
		conv_dim = image_dim - filter_dim + 1;
		feature = arma::zeros(conv_dim,conv_dim,sample_num);
		filter = rot(W,2);
	}
#ifdef OPENMP
#pragma omp parallel for shared(conv_dim,full_image,filter_dim,feature)
#endif
	for(int i = 0;i < conv_dim; i++){
		for(int j = 0;j < conv_dim;j++){
			patch = full_image.tube(i,j,i+filter_dim-1,j+filter_dim-1);
			feature_map = patch % filter;
			for(int k = 0;k < sample_num; k++){
				feature(i,j,k) = arma::sum(arma::sum(feature_map.slice(k)));
			}
		}
	}
	

	return feature;
}
//TODO: 实现convn的full和conv2
#endif