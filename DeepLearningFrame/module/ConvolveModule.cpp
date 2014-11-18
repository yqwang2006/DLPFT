#include "ConvolveModule.h"
#include "../util/convolve.h"
using namespace dlpft::module;
void ConvolveModule::initial_weights_bias(){
	if(load_weight == "YES"){
		if(weight_addr != "" && bias_addr != ""){
			if(initial_weights_bias_from_file(weight_addr,bias_addr)){
				return;
			}
		}
	}
		bias = zeros(outputImageNum,1);
		weightMatrix = zeros(filterDim*filterNum,filterDim);

		cube tempW = 0.1 * randn(filterDim,filterDim,filterNum);
		for(int i = 0; i < filterNum; i++){
			weightMatrix.rows(i*filterDim,(i+1)*filterDim-1) = tempW.slice(i);
		}
	
	
}

arma::mat ConvolveModule::forwardpropagate(const arma::mat data,  NewParam param){
	

	const int samples_num = data.n_cols;
	int outputMapSize = outputImageDim * outputImageDim;

	arma::mat all_features = arma::zeros(outputMapSize*outputImageNum,samples_num);
	cube features_filter = zeros(outputImageDim,outputImageDim,samples_num);
	mat W = zeros(filterDim,filterDim);
	mat images = zeros(inputImageDim*inputImageDim,samples_num);
	cube all_images = zeros(inputImageDim*inputImageDim,samples_num,1);
//#ifdef OPENMP
//#pragma omp parallel for private(features_filter,W,images,all_images) shared(data,all_features)
//#endif
	for(int nout = 0; nout < outputImageNum; nout ++){
			features_filter = zeros(outputImageDim,outputImageDim,samples_num);
			int fmInBase = 0;
			double b = (bias.row(nout))(0);
			for(int nin = 0; nin < inputImageNum; nin++){
				W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				images = data.rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1);
				all_images = zeros(inputImageDim*inputImageDim,samples_num,1);
				all_images.slice(0) = images;
				all_images.reshape(inputImageDim,inputImageDim,samples_num);
				//reshape(images,inputImageDim,inputImageDim,inputImageNum);
				features_filter += convn_cube(all_images,W,"valid");
			}
			features_filter = features_filter + b;
			//features_filter = active_function(activeFuncChoice,features_filter);
			features_filter.reshape(outputMapSize,samples_num,1);
			arma::mat& temp_filter = features_filter.slice(0);
			all_features.rows(nout*outputMapSize,(nout+1)*outputMapSize-1) = temp_filter;
		}

	all_features = active_function(activeFuncChoice,all_features);
	return all_features;
}
arma::mat ConvolveModule::process_delta(arma::mat curr_delta){
	const int samples_num = curr_delta.n_cols;
	int delta_dim = sqrt(curr_delta.n_rows / outputImageNum);
	arma::mat convn_delta = zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	//int conv_dim = delta_dim - filter_dim + 1;

	//arma::mat curr_delta = arma::zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	mat W = zeros(filterDim,filterDim);
	mat single_delta = zeros(delta_dim*delta_dim,samples_num);
	arma::cube all_deltas = zeros(delta_dim*delta_dim,samples_num,1);
	arma::cube delta_filter = zeros(inputImageDim,inputImageDim,samples_num);
		for(int nin = 0; nin < inputImageNum; nin++){
			int fmInBase = 0;
			delta_filter = zeros(inputImageDim,inputImageDim,samples_num);
			for(int nout = 0; nout < outputImageNum; nout ++){
				double b = (bias.row(nout))(0);
				W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				
				single_delta = curr_delta.rows(nout*delta_dim*delta_dim,(nout+1)*delta_dim*delta_dim-1);
				all_deltas = zeros(delta_dim*delta_dim,samples_num,1);
				all_deltas.slice(0) = single_delta;
				all_deltas.reshape(delta_dim,delta_dim,samples_num);
				delta_filter += convn_cube(all_deltas,W,"full");
			}
			delta_filter.reshape(inputImageDim*inputImageDim,samples_num,1);
			arma::mat& temp_delta = delta_filter.slice(0);
			convn_delta.rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1) =temp_delta;
		}
		
	
	return convn_delta;
}

arma::mat ConvolveModule::backpropagate(const arma::mat next_delta, const arma::mat features, NewParam param){
	//卷基层的下一层一般是pooling层，pooling层和当前的卷基层输出maps个数一样
	//同时pooling层对每个map乘以一个常数beta(即下面代码中的weightMatrix(i)，和bias(i)
	//之后再输出多个maps
	clock_t start_time = clock();
	clock_t end_time;
	double duration = 0;


	arma::mat curr_delta = zeros(outputSize,next_delta.n_cols);
	int outputImageSize = outputImageDim*outputImageDim;

	curr_delta = active_function_dev(activeFuncChoice,features) % next_delta;

	return curr_delta;
	

}
void ConvolveModule::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta, NewParam param,double weight_decay,arma::mat& Wgrad, arma::mat& bgrad){
	//compute bgrad
	int mbSize = input_data.n_cols;
	double lambda = 3e-3;
	Wgrad.set_size(filterDim*filterNum,filterDim);
	bgrad.set_size(outputImageNum,1);
	mat Wgrad_j_i = zeros(filterDim,filterDim);
	mat input_images = zeros(inputImageDim*inputImageDim,mbSize);
	mat delta_i_k = zeros(outputImageDim*outputImageDim,mbSize);
	cube all_images = arma::zeros(inputImageDim*inputImageDim,mbSize,1);
	cube all_delta = zeros(outputImageDim*outputImageDim,mbSize,1);
	cube temp_wgrad;
	for(int i = 0; i < outputImageNum; i++){
		for(int j = 0;j < inputImageNum;j++){
			Wgrad_j_i = zeros(filterDim,filterDim);
		
			input_images = input_data.rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
			delta_i_k = delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1);

			all_images = zeros(inputImageDim*inputImageDim,mbSize,1);
			all_images.slice(0) = input_images;
			all_images.reshape(inputImageDim,inputImageDim,mbSize);

			all_delta = zeros(outputImageDim*outputImageDim,mbSize,1);
			all_delta.slice(0) = delta_i_k;
			all_delta.reshape(outputImageDim,outputImageDim,mbSize);

			temp_wgrad = convn_cube(all_images,all_delta,"valid");

			for(int k = 0;k < temp_wgrad.n_slices; k++){
				Wgrad_j_i += temp_wgrad.slice(k);
			}
			
			
			Wgrad.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum+j+1)-1) 
				= ((double)1/mbSize)*Wgrad_j_i
				+lambda*weightMatrix.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum + j + 1)-1);
		}

		bgrad(i) = ((double)1/mbSize)*accu(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1));
	}

}