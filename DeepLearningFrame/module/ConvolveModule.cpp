#include "ConvolveModule.h"
#include "../util/convolve.h"
using namespace dlpft::module;
void ConvolveModule::initial_weights_bias(){
	bias = zeros(outputImageNum,1);
	weightMatrix = zeros(filterDim*filterNum,filterDim);
#if DEBUG
	for(int i = 0; i < filterNum; i++){
		weightMatrix.rows(i*filterDim,(i+1)*filterDim-1) = (i+1)*0.1*ones(filterDim,filterDim);
	}
#else
	cube tempW = 0.1 * randn(filterDim,filterDim,filterNum);
	for(int i = 0; i < filterNum; i++){
		weightMatrix.rows(i*filterDim,(i+1)*filterDim-1) = tempW.slice(i);
	}
#endif
	
	
}
arma::mat ConvolveModule::forwardpropagate(const arma::mat data,  NewParam param){
	const int samples_num = data.n_cols;

	arma::mat all_features = arma::zeros(outputImageDim*outputImageDim*outputImageNum,samples_num);
	mat features_filter = zeros(outputImageDim,outputImageDim);
	mat W = zeros(filterDim,filterDim);
	mat image = zeros(inputImageDim*inputImageDim,1);
	for(int n = 0;n < samples_num; n++){
		
		
		for(int nout = 0; nout < outputImageNum; nout ++){
			features_filter = zeros(outputImageDim,outputImageDim);
			int fmInBase = 0;
			double b = (bias.row(nout))(0);
			for(int nin = 0; nin < inputImageNum; nin++){
				W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				image = data.col(n).rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1);
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
		
		for(int nin = 0; nin < inputImageNum; nin++){
			int fmInBase = 0;
			arma::mat delta_filter = zeros(inputImageDim,inputImageDim);
			for(int nout = 0; nout < outputImageNum; nout ++){
				double b = (bias.row(nout))(0);
				arma::mat W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				
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
	//卷基层的下一层一般是pooling层，pooling层和当前的卷基层输出maps个数一样
	//同时pooling层对每个map乘以一个常数beta(即下面代码中的weightMatrix(i)，和bias(i)
	//之后再输出多个maps
	
	arma::mat curr_delta = zeros(outputSize,next_delta.n_cols);
	for(int i = 0;i < outputImageNum;i ++){
//#if DEBUG
		curr_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1) 
			= (active_function_dev(activeFuncChoice,features.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)) 
			% next_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1));
//#else 
//		curr_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1) 
//			= next_layer_weight(i)
//			*(active_function_dev(activeFuncChoice,features.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)) 
//			% next_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1));
//#endif
	}
	
	return curr_delta;
	

}
void ConvolveModule::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta, NewParam param,arma::mat& Wgrad, arma::mat& bgrad){
	//compute bgrad
	int mbSize = input_data.n_cols;
	double lambda = 3e-3;
	Wgrad.set_size(filterDim*filterNum,filterDim);
	bgrad.set_size(outputImageNum,1);
	
	for(int i = 0; i < outputImageNum; i++){
		for(int j = 0;j < inputImageNum;j++){
			mat Wgrad_j_i = zeros(filterDim,filterDim);
			for(int k = 0;k < mbSize; k++){
				mat input_image = input_data.col(k).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
				mat delta_i_k = delta.col(k).rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1);

				input_image.reshape(inputImageDim,inputImageDim);
				delta_i_k.reshape(outputImageDim,outputImageDim);
				Wgrad_j_i += convn(input_image,delta_i_k,"valid");
			}
			Wgrad.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum+j+1)-1) = ((double)1/mbSize)*Wgrad_j_i
				+lambda*weightMatrix.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum + j + 1)-1);
		}

		bgrad(i) = ((double)1/mbSize)*sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
	}
}