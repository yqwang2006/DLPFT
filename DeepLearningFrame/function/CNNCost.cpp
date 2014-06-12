#include "CNNCost.h"
#include "../util/onehot.h"
using namespace dlpft::module;
using namespace dlpft::param;
using namespace dlpft::function;
void CNNCost::initialParam(){
}
double CNNCost::value_gradient(arma::mat& grad){
	int image_dim = sqrt(data.n_rows);
	int num_images = data.n_cols;
	double lambda = 3e-3;
	arma::mat *delta = new arma::mat[layer_num+1];
	grad = zeros(coefficient.size(),1);
	cnnParamsToStack();

	arma::mat* activations = new arma::mat[layer_num];
	ResultModel result_model;
	double cost = 0;
	//forward Propagation
	int start_b_loc = 0;
	for(int i = 0;i < layer_num;i ++){
		if(i == 0){
			activations[i] = modules[i]->forwardpropagate(data,params[i]);
		}else{
			activations[i] = modules[i]->forwardpropagate(activations[i-1],params[i]);
		}
		cost += (lambda/2)*arma::sum(arma::sum(arma::pow(modules[i]->weightMatrix,2)));
		start_b_loc += modules[i]->weightMatrix.size();
	}


	arma::mat desired_out = onehot(activations[layer_num-1].n_rows,activations[layer_num-1].n_cols,labels);

	arma::mat gm = desired_out.t()*arma::log(reshape(activations[layer_num-1],activations[layer_num-1].size(),1));

	cost += ((double)-1/num_images)*gm(0);

	//backward propagation to compute delta

	delta[layer_num] = (desired_out - activations[layer_num-1]);
	arma::mat next_layer_weight;
	for(int i = layer_num-1;i >=0 ;i--){
		arma::mat w_grad = zeros(modules[i]->weightMatrix.n_rows,modules[i]->weightMatrix.n_cols);
		arma::mat b_grad = zeros(modules[i]->bias.size(),1);
		if(i == layer_num-1){
			delta[i] = modules[i]->backpropagate(next_layer_weight,delta[i+1],activations[i],params[i]);
			w_grad = ((double)1/num_images)*delta[i] * data.t();
			
		}else{
			delta[i] = modules[i]->backpropagate(modules[i+1]->weightMatrix,delta[i+1],activations[i],params[i]);
			w_grad = ((double)1/num_images)*delta[i] * activations[i-1].t();
		}
		b_grad = ((double)1/num_images)*arma::sum(delta[i],1);

	}
	

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
		int rows_num = 0,cols_num = 0;
		if(params[i].params[params_name[ALGORITHM]] == "ConvolveModule"){
			int number_filters = ((ConvolveModule*) modules[i])->filterNum;
			int filter_dim = ((ConvolveModule*) modules[i])->filterDim;
			rows_num = filter_dim*number_filters;
			cols_num = filter_dim;

		}else if(params[i].params[params_name[ALGORITHM]] == "FullConnection"){
			rows_num = ((FullConnectModule*) modules[i])->inputSize;
			cols_num = ((FullConnectModule*) modules[i])->outputSize;
		}else if(params[i].params[params_name[ALGORITHM]] == "SoftMax"){
			rows_num = ((SoftMax*) modules[i])->inputSize;
			cols_num = ((SoftMax*) modules[i])->outputSize;

		}
		end_w_loc += modules[i]->weightMatrix.size();
		modules[i]->weightMatrix = arma::reshape(coefficient.rows(start_w_loc,end_w_loc-1),rows_num,cols_num);
		start_w_loc = end_w_loc;
	}
	start_b_loc = end_w_loc;
	end_b_loc = end_w_loc;
	for(int i = 0;i < layer_num; i++){
		int hiddenSize = 0;

		end_b_loc += modules[i]->bias.size();
		arma::mat b = arma::reshape(coefficient.rows(start_b_loc,end_b_loc-1),end_b_loc-start_b_loc,1);
		modules[i]->bias = b;

		start_b_loc = end_b_loc;
	}
}