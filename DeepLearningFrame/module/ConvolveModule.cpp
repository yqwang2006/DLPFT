#include "ConvolveModule.h"
#include "../util/convolve.h"
using namespace dlpft::module;
void ConvolveModule::initial_weights_bias(){
	bias = zeros(outputImageDim,1);
	weightMatrix = 0.1 * randu<arma::mat> (filterDim*filterNum,filterDim);
}
arma::mat ConvolveModule::forwardpropagate(const arma::mat data,  NewParam param){
	const int samples_num = data.n_cols;

	arma::mat all_features = arma::zeros(outputImageDim*outputImageDim*filterDim,samples_num);
	
	for(int n = 0;n < samples_num; n++){
		arma::mat features_filter = zeros(outputImageDim,outputImageDim);
		for(int nout = 0; nout < outputImageNum; nout ++){
			int fmInBase = 0;
			double b = (bias.row(nout))(0);
			for(int nin = 0; nin < inputImageNum; nin++){
				arma::mat W = weightMatrix.rows(filterDim * (nout*outputImageNum + nin),filterDim * (nout*outputImageNum + nin + 1)-1);
				arma::mat image = data.col(n).rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1);
				image.reshape(inputImageDim,inputImageDim);
				features_filter += convn(image,W,"valid");
			}
			features_filter = features_filter + b;
			features_filter = active_function(activeFuncChoice,features_filter);
			all_features.col(n).rows(nout*features_filter.size(),(nout+1)*features_filter.size()-1) = reshape(features_filter,features_filter.size(),1);
		}
		
	}

	return all_features;
}
arma::mat ConvolveModule::process_delta(arma::mat curr_delta){
	const int samples_num = curr_delta.n_cols;
	int delta_dim = sqrt(curr_delta.n_rows / outputImageNum);
	arma::mat convn_delta = zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	//int conv_dim = delta_dim - filter_dim + 1;

	//arma::mat curr_delta = arma::zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	
	for(int n = 0;n < samples_num; n++){
		arma::mat delta_filter = zeros(inputImageDim,inputImageDim);
		for(int nin = 0; nin < inputImageNum; nin++){
			int fmInBase = 0;
			
			for(int nout = 0; nout < outputImageNum; nout ++){
				double b = (bias.row(nout))(0);
				arma::mat W = weightMatrix.rows(filterDim * (nout*outputImageNum + nin),filterDim * (nout*outputImageNum + nin + 1)-1);
				
				arma::mat single_delta = curr_delta.col(n).rows(nout*delta_dim*delta_dim,(nout+1)*delta_dim*delta_dim-1);
				single_delta.reshape(delta_dim,delta_dim);
				delta_filter += convn(single_delta,W,"full");
			}
			convn_delta.col(n).rows(nin*delta_filter.size(),(nin+1)*delta_filter.size()-1) = reshape(delta_filter,delta_filter.size(),1);
		}
		
	}
	return convn_delta;
}
arma::mat ConvolveModule::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	arma::mat curr_delta = zeros(outputSize,next_delta.n_cols);
	for(int i = 0;i < outputImageNum;i ++){
		curr_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1) 
			= next_layer_weight(i)*(active_function_dev(activeFuncChoice,features.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)) 
			% next_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1));
	}
	
	return curr_delta;
	

}
