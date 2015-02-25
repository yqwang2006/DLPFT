#ifndef CONVOLVEFFT_H
#define CONVOLVEFFT_H

#include "armadillo"
#include "convolve.h"

static arma::mat convnfft(const arma::mat image, const arma::mat W, std::string info){
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
	int fftsize = image_dim + filter_dim - 1;
	arma::cx_mat tmpimg = arma::fft2(full_image,fftsize,fftsize);
	arma::cx_mat tmpfilter = arma::fft2(filter,fftsize,fftsize);

	arma::mat tfeature = arma::real(arma::ifft2(tmpimg % tmpfilter));
	feature = tfeature.submat(0,0,conv_dim-1,conv_dim-1);
	return feature;
}
static arma::cube convnfft_cube(const arma::cube images, const arma::mat W, std::string info){
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
	int fftsize = image_dim + filter_dim - 1;
	arma::cx_mat tmpimg;
	arma::cx_mat tmpfilter;
	arma::mat tfeature = arma::zeros(fftsize,fftsize);
	for(int i = 0;i < sample_num; i++){
		
		tmpimg = arma::fft2(full_image.slice(i),fftsize,fftsize);
		tmpfilter = arma::fft2(filter,fftsize,fftsize);
		tfeature = arma::real(arma::ifft2(tmpimg % tmpfilter));
		feature.slice(i) = tfeature.submat(0,0,conv_dim-1,conv_dim-1);
	}

	return feature;
}
static arma::cube convnfft_cube(const arma::cube images, const arma::cube W, std::string info){
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

	for(int i = 0;i < sample_num; i++){
		int fftsize = image_dim + filter_dim - 1;
		arma::cx_mat tmpimg = arma::fft2(full_image.slice(i),fftsize,fftsize);
		arma::cx_mat tmpfilter = arma::fft2(filter.slice(i),fftsize,fftsize);
		arma::mat tfeature = arma::real(arma::ifft2(tmpimg % tmpfilter));
		feature.slice(i) = tfeature.submat(0,0,conv_dim-1,conv_dim-1);
	}

	return feature;
}
//TODO: 实现convn的full和conv2
#endif