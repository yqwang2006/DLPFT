#include "ModelCost.h"
#include "../util/onehot.h"
using namespace dlpft::module;
using namespace dlpft::param;
using namespace dlpft::function;
void ModelCost::initialParam(){
}
double ModelCost::value_gradient(arma::mat& grad){
	clock_t start_time = clock();
	clock_t end_time;
	double duration = 0;


	int image_dim = sqrt(data.n_rows);
	int num_images = data.n_cols;
	double lambda = 3e-3;
	arma::mat *delta = new arma::mat[layer_num+1];
	grad = zeros(coefficient.size(),1);
	paramsToStack();

	arma::mat* activations = new arma::mat[layer_num];
	double cost = 0;
	//forward Propagation
	
	
	for(int i = 0;i < layer_num;i ++){

		if(i == 0){
			activations[i] = modules[i]->forwardpropagate(data,params[i]);
		}else{

			activations[i] = modules[i]->forwardpropagate(activations[i-1],params[i]);
		}

		/*ofstream ofs;
		ofs.open("weightMat.txt");
		modules[i]->weightMatrix.quiet_save(ofs,raw_ascii);
		ofs.close();*/

	}
	arma::mat desired_out;
	if(modules[layer_num-1]->name == "SoftMax"){

		desired_out = onehot(activations[layer_num-1].n_rows,activations[layer_num-1].n_cols,labels);
		cost += (lambda/2)*arma::sum(arma::sum(arma::pow(modules[layer_num-1]->weightMatrix,2)));

	}
	else
		desired_out = labels.t();
	//arma::mat gm = reshape(desired_out,desired_out.size(),1).t()*log(reshape(activations[layer_num-1],activations[layer_num-1].size(),1));

	double gm_cost = dot(reshape(desired_out,desired_out.size(),1),log(reshape(activations[layer_num-1],activations[layer_num-1].size(),1)));

	cost += ((double)-1/num_images)*gm_cost;
	

	//backward propagation to compute delta

	delta[layer_num] = -(desired_out - activations[layer_num-1]);
	arma::mat next_delta = delta[layer_num];
	arma::mat input_data,next_layer_weight;
	
	int curr_loc = coefficient.size()-1;
	for(int i = layer_num-1;i >=0 ;i--){
		arma::mat w_grad = zeros(modules[i]->weightMatrix.n_rows,modules[i]->weightMatrix.n_cols);
		arma::mat b_grad = zeros(modules[i]->bias.size(),1);
		
		if(layer_num == 1){
			input_data = data;
			next_layer_weight = zeros(modules[i]->weightMatrix.size(),1);
		}
		else if(i == layer_num-1){
			input_data = activations[i-1];
		}
		else if(i == 0){
			input_data = data;
			next_layer_weight = modules[i+1]->weightMatrix;

			
		}
		else{
			input_data = activations[i-1];
			next_layer_weight = modules[i+1]->weightMatrix;

		}

		delta[i] = modules[i]->backpropagate(next_layer_weight,next_delta,activations[i],params[i]);
		modules[i]->calculate_grad_using_delta(input_data,delta[i],params[i],weight_decay,w_grad,b_grad);

		if(i > 0){
			next_delta = modules[i]->process_delta(delta[i]);
		}

		grad.rows(curr_loc - b_grad.size()+1, curr_loc) = reshape(b_grad,b_grad.size(),1);
		grad.rows(curr_loc-b_grad.size()-w_grad.size()+1,curr_loc-b_grad.size()) = reshape(w_grad,w_grad.size(),1);
		
		curr_loc = curr_loc - w_grad.size() - b_grad.size();

	}


	delete[] activations;
	delete[] delta;
	return cost;
}
void ModelCost::gradient(arma::mat& grad){

}
void ModelCost::hessian(arma::mat& grad, arma::mat& hess){

}
void ModelCost::paramsToStack(){

	int curr_loc = 0;
	for(int i = 0;i < layer_num; i++){
		int hiddenSize = 0;
		int rows_num = 0,cols_num = 0;
		if(params[i].params[params_name[ALGORITHM]] == "ConvolveModule" ){
			int number_filters = ((ConvolveModule*) modules[i])->filterNum;
			int filter_dim = ((ConvolveModule*) modules[i])->filterDim;
			rows_num = filter_dim*number_filters;
			cols_num = filter_dim;

		}else if(params[i].params[params_name[ALGORITHM]] == "CRBM" ){
			int number_filters = ((ConvolutionRBM*) modules[i])->filterNum;
			int filter_dim = ((ConvolutionRBM*) modules[i])->filterDim;
			rows_num = filter_dim*number_filters;
			cols_num = filter_dim;

		}else if(params[i].params[params_name[ALGORITHM]] == "Pooling"){
			rows_num = ((Pooling*) modules[i])->outputImageNum;
			cols_num = 1;
		}else{
			rows_num = modules[i]->outputSize;
			cols_num = modules[i]->inputSize;
		}
		
		modules[i]->weightMatrix = arma::reshape(coefficient.rows(curr_loc,curr_loc + modules[i]->weightMatrix.size()-1),rows_num,cols_num);
		
		curr_loc += modules[i]->weightMatrix.size();

		arma::mat b = arma::reshape(coefficient.rows(curr_loc,curr_loc+modules[i]->bias.size()-1),modules[i]->bias.size(),1);
		modules[i]->bias = b;
		curr_loc += modules[i]->bias.size();
		
	}
}