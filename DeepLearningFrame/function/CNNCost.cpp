#include "CNNCost.h"
using namespace dlpft::module;
using namespace dlpft::param;
using namespace dlpft::function;
void CNNCost::initialParam(){
}
double CNNCost::value_gradient(arma::mat& grad){
	int image_dim = sqrt(data.n_rows);
	int num_images = data.n_cols;
	double lambda = 3e-3;

	cnnParamsToStack();
	
	arma::mat* activations = new arma::mat[layer_num];
	ResultModel result_model;

	//forward Propagation
	for(int i = 0;i < layer_num;i ++){
		if(i == 0){
			activations[i] = modules[i]->forwardpropagate(data,params[i]);
		}else{
			activations[i] = modules[i]->forwardpropagate(activations[i-1],params[i]);
		}
	}
	double cost = 0;




	delete[] activations;
	return cost;
}
void CNNCost::gradient(arma::mat& grad){

}
void CNNCost::hessian(arma::mat& grad, arma::mat& hess){

}
void CNNCost::cnnParamsToStack(){
	
	int start_w_loc = 0;
	int start_b_loc = 0;
	int end_w_loc = 0;
	int end_b_loc = 0;
	for(int i = 0;i < layer_num; i++){
		int hiddenSize = 0;
		if(params[i].params[params_name[ALGORITHM]] == "ConvolveModule"){
			int number_filters = ((ConvolveModule*) modules[i])->filterNum;
			int filter_dim = ((ConvolveModule*) modules[i])->filterDim;
			end_w_loc += ((ConvolveModule*) modules[i])->weightMatrix.size();
			arma::mat W = arma::reshape(coefficient.rows(start_w_loc,end_w_loc-1),filter_dim*number_filters,filter_dim);
			start_w_loc = end_w_loc;
			modules[i]->weightMatrix = W;
		}else if(params[i].params[params_name[ALGORITHM]] == "FullConnection"){
			int inputSize = ((FullConnectModule*) modules[i])->inputSize;
			int outputSize = ((FullConnectModule*) modules[i])->outputSize;
			end_w_loc += ((FullConnectModule*) modules[i])->weightMatrix.size();
			arma::mat W = arma::reshape(coefficient.rows(start_w_loc,end_w_loc-1),inputSize,outputSize);
			start_w_loc = end_w_loc;
			modules[i]->weightMatrix = W;
		}else if(params[i].params[params_name[ALGORITHM]] == "SoftMax"){
			int inputSize = ((SoftMax*) modules[i])->inputSize;
			int outputSize = ((SoftMax*) modules[i])->outputSize;
			end_w_loc += ((SoftMax*) modules[i])->weightMatrix.size();
			arma::mat W = arma::reshape(coefficient.rows(start_w_loc,end_w_loc-1),inputSize,outputSize);
			start_w_loc = end_w_loc;
			modules[i]->weightMatrix = W;
		}
	}
	start_b_loc = end_w_loc;
	end_b_loc = end_w_loc;
	for(int i = 0;i < layer_num; i++){
		int hiddenSize = 0;
		if(params[i].params[params_name[ALGORITHM]] == "ConvolveModule"){
			end_b_loc += ((ConvolveModule*) modules[i])->bias.size();
			arma::mat b = arma::reshape(coefficient.rows(start_b_loc,end_b_loc-1),end_b_loc-start_b_loc,1);
			((ConvolveModule*) modules[i])->bias = b;
		}else if(params[i].params[params_name[ALGORITHM]] == "FullConnection"){
			end_b_loc += ((FullConnectModule*) modules[i])->bias.size();
			arma::mat b = arma::reshape(coefficient.rows(start_b_loc,end_b_loc-1),end_b_loc-start_b_loc,1);
			((FullConnectModule*) modules[i])->bias = b;
		}else if(params[i].params[params_name[ALGORITHM]] == "SoftMax"){
			end_b_loc += ((SoftMax*) modules[i])->bias.size();
			arma::mat b = arma::reshape(coefficient.rows(start_b_loc,end_b_loc-1),end_b_loc-start_b_loc,1);
			((SoftMax*) modules[i])->bias = b;
		}
		start_b_loc = end_b_loc;
	}
}