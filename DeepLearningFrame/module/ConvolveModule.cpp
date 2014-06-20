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
//arma::mat ConvolveModule::forwardpropagate(const arma::mat data,  NewParam param){
//	clock_t start_time = clock();
//	clock_t end_time;
//	double duration = 0;
//	
//	
//	const int samples_num = data.n_cols;
//
//	arma::mat all_features = arma::zeros(outputImageDim*outputImageDim*outputImageNum,samples_num);
//	mat features_filter = zeros(outputImageDim,outputImageDim);
//	mat W = zeros(filterDim,filterDim);
//	mat image = zeros(inputImageDim*inputImageDim,1);
//	for(int n = 0;n < samples_num; n++){
//		
//		
//		for(int nout = 0; nout < outputImageNum; nout ++){
//			features_filter = zeros(outputImageDim,outputImageDim);
//			int fmInBase = 0;
//			double b = (bias.row(nout))(0);
//			for(int nin = 0; nin < inputImageNum; nin++){
//				W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
//				image = data.col(n).rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1);
//				image.reshape(inputImageDim,inputImageDim);
//				features_filter += convn(image,W,"valid");
//			}
//			features_filter = features_filter + b;
//			features_filter = active_function(activeFuncChoice,features_filter);
//			all_features.col(n).rows(nout*features_filter.size(),(nout+1)*features_filter.size()-1) = reshape(features_filter,features_filter.size(),1);
//		}
//		
//	}
//  end_time = clock();
//  duration = (double)(end_time-start_time)/CLOCKS_PER_SEC;
//	cout << "convolve forward spent: " << duration << " s" << endl;
//	return all_features;
//}

arma::mat ConvolveModule::forwardpropagate(const arma::mat data,  NewParam param){
	

	const int samples_num = data.n_cols;
	int outputMapSize = outputImageDim * outputImageDim;

	arma::mat all_features = arma::zeros(outputMapSize*outputImageNum,samples_num);
	cube features_filter = zeros(outputImageDim,outputImageDim,samples_num);
	mat W = zeros(filterDim,filterDim);
	mat images = zeros(inputImageDim*inputImageDim,samples_num);

		for(int nout = 0; nout < outputImageNum; nout ++){
			features_filter = zeros(outputImageDim,outputImageDim,samples_num);
			int fmInBase = 0;
			double b = (bias.row(nout))(0);
			for(int nin = 0; nin < inputImageNum; nin++){
				W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				images = data.rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1);
				cube all_images = zeros(inputImageDim*inputImageDim,samples_num,1);
				all_images.slice(0) = images;
				all_images.reshape(inputImageDim,inputImageDim,samples_num);
				//reshape(images,inputImageDim,inputImageDim,inputImageNum);
				
				features_filter += convn_cube(all_images,W,"valid");
			}
			features_filter = features_filter + b;
			//features_filter = active_function(activeFuncChoice,features_filter);
			features_filter.reshape(outputMapSize,samples_num,1);
			arma::mat temp_filter = features_filter.slice(0);
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
	
		
		for(int nin = 0; nin < inputImageNum; nin++){
			int fmInBase = 0;
			arma::cube delta_filter = zeros(inputImageDim,inputImageDim,samples_num);
			for(int nout = 0; nout < outputImageNum; nout ++){
				double b = (bias.row(nout))(0);
				arma::mat W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				
				arma::mat single_delta = curr_delta.rows(nout*delta_dim*delta_dim,(nout+1)*delta_dim*delta_dim-1);
				arma::cube all_deltas = zeros(delta_dim*delta_dim,samples_num,1);
				all_deltas.slice(0) = single_delta;
				all_deltas.reshape(delta_dim,delta_dim,samples_num);
				delta_filter += convn_cube(all_deltas,W,"full");
			}
			delta_filter.reshape(inputImageDim*inputImageDim,samples_num,1);
			arma::mat temp_delta = delta_filter.slice(0);
			convn_delta.rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1) =temp_delta;
		}
		
	
	return convn_delta;
}
//arma::mat ConvolveModule::process_delta(arma::mat curr_delta){
//	const int samples_num = curr_delta.n_cols;
//	int delta_dim = sqrt(curr_delta.n_rows / outputImageNum);
//	arma::mat convn_delta = zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
//	//int conv_dim = delta_dim - filter_dim + 1;
//
//	//arma::mat curr_delta = arma::zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
//	
//	for(int n = 0;n < samples_num; n++){
//		
//		for(int nin = 0; nin < inputImageNum; nin++){
//			int fmInBase = 0;
//			arma::mat delta_filter = zeros(inputImageDim,inputImageDim);
//			for(int nout = 0; nout < outputImageNum; nout ++){
//				double b = (bias.row(nout))(0);
//				arma::mat W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
//				
//				arma::mat single_delta = curr_delta.col(n).rows(nout*delta_dim*delta_dim,(nout+1)*delta_dim*delta_dim-1);
//				single_delta.reshape(delta_dim,delta_dim);
//				delta_filter += convn(single_delta,W,"full");
//			}
//			convn_delta.col(n).rows(nin*delta_filter.size(),(nin+1)*delta_filter.size()-1) = reshape(delta_filter,delta_filter.size(),1);
//		}
//		
//	}
//	return convn_delta;
//}
arma::mat ConvolveModule::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	//��������һ��һ����pooling�㣬pooling��͵�ǰ�ľ�������maps����һ��
	//ͬʱpooling���ÿ��map����һ������beta(����������е�weightMatrix(i)����bias(i)
	//֮����������maps
	clock_t start_time = clock();
	clock_t end_time;
	double duration = 0;


	arma::mat curr_delta = zeros(outputSize,next_delta.n_cols);
	int outputImageSize = outputImageDim*outputImageDim;

	curr_delta = active_function_dev(activeFuncChoice,features) % next_delta;


//	for(int i = 0;i < outputImageNum;i ++){
////#if DEBUG
//		curr_delta.rows(i*outputImageSize,(i+1)*outputImageSize-1) 
//			= active_function_dev(activeFuncChoice,features.rows(i*outputImageSize,(i+1)*outputImageSize-1)) 
//			% next_delta.rows(i*outputImageSize,(i+1)*outputImageSize-1);
////#else 
////		curr_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1) 
////			= next_layer_weight(i)
////			*(active_function_dev(activeFuncChoice,features.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)) 
////			% next_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1));
////#endif
//	}
	//  end_time = clock();
 //  duration = (double)(end_time-start_time)/CLOCKS_PER_SEC;
	//cout << "convolve backward spent: " << duration << " s" << endl;
	return curr_delta;
	

}
//void ConvolveModule::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta, NewParam param,arma::mat& Wgrad, arma::mat& bgrad){
//	//compute bgrad
//	clock_t start_time = clock();
//	clock_t end_time;
//	double duration = 0;
//	
//	int mbSize = input_data.n_cols;
//	double lambda = 3e-3;
//	Wgrad.set_size(filterDim*filterNum,filterDim);
//	bgrad.set_size(outputImageNum,1);
//	
//	for(int i = 0; i < outputImageNum; i++){
//		for(int j = 0;j < inputImageNum;j++){
//			mat Wgrad_j_i = zeros(filterDim,filterDim);
//			for(int k = 0;k < mbSize; k++){
//				mat input_image = input_data.col(k).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
//				mat delta_i_k = delta.col(k).rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1);
//
//				input_image.reshape(inputImageDim,inputImageDim);
//				delta_i_k.reshape(outputImageDim,outputImageDim);
//				Wgrad_j_i += convn(input_image,delta_i_k,"valid");
//			}
//			Wgrad.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum+j+1)-1) = ((double)1/mbSize)*Wgrad_j_i
//				+lambda*weightMatrix.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum + j + 1)-1);
//		}
//
//		bgrad(i) = ((double)1/mbSize)*sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
//	}
//
//	end_time = clock();
//    duration = (double)(end_time-start_time)/CLOCKS_PER_SEC;
//	cout << "convolve grad compute spent: " << duration << " s" << endl;
//}
void ConvolveModule::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta, NewParam param,arma::mat& Wgrad, arma::mat& bgrad){
	//compute bgrad
	clock_t start_time = clock();
	clock_t end_time;
	double duration = 0;
	
	int mbSize = input_data.n_cols;
	double lambda = 3e-3;
	Wgrad.set_size(filterDim*filterNum,filterDim);
	bgrad.set_size(outputImageNum,1);
	
	for(int i = 0; i < outputImageNum; i++){
		for(int j = 0;j < inputImageNum;j++){
			mat Wgrad_j_i = zeros(filterDim,filterDim);
		
			mat input_images = input_data.rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
			mat delta_i_k = delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1);

			cube all_images = arma::zeros(inputImageDim*inputImageDim,mbSize,1);
			all_images.slice(0) = input_images;
			all_images.reshape(inputImageDim,inputImageDim,mbSize);

			cube all_delta = zeros(outputImageDim*outputImageDim,mbSize,1);
			all_delta.slice(0) = delta_i_k;
			all_delta.reshape(outputImageDim,outputImageDim,mbSize);

			cube temp_wgrad = convn_cube(all_images,all_delta,"valid");

			for(int k = 0;k < temp_wgrad.n_slices; k++){
				Wgrad_j_i += temp_wgrad.slice(k);
			}
			
			
			Wgrad.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum+j+1)-1) 
				= ((double)1/mbSize)*Wgrad_j_i;
				//+lambda*weightMatrix.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum + j + 1)-1);
		}

		bgrad(i) = ((double)1/mbSize)*sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
	}

	//end_time = clock();
 //   duration = (double)(end_time-start_time)/CLOCKS_PER_SEC;
	//cout << "convolve grad compute spent: " << duration << " s" << endl;
}