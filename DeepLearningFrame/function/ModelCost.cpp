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
	arma::mat gm = reshape(desired_out,desired_out.size(),1).t()*log(reshape(activations[layer_num-1],activations[layer_num-1].size(),1));

	

	cost += ((double)-1/num_images)*gm(0);

	//backward propagation to compute delta

	delta[layer_num] = -(desired_out - activations[layer_num-1]);
	int start_w_loc=0,end_w_loc=start_b_loc,end_b_loc=grad.size();
	arma::mat next_delta = delta[layer_num];
	arma::mat input_data,next_layer_weight;
	

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
		modules[i]->calculate_grad_using_delta(input_data,delta[i],params[i],w_grad,b_grad);

		
		if(i > 0)
			next_delta = modules[i]->process_delta(delta[i]);
		
		start_w_loc = end_w_loc - modules[i]->weightMatrix.size();
		start_b_loc = end_b_loc - modules[i]->bias.size();
		grad.rows(start_w_loc,end_w_loc-1) = reshape(w_grad,w_grad.size(),1);
		grad.rows(start_b_loc,end_b_loc-1) = reshape(b_grad,b_grad.size(),1);
		end_w_loc -= modules[i]->weightMatrix.size();
		end_b_loc -= modules[i]->bias.size();

	}

	

	//end_time = clock();
 //   duration = (double)(end_time-start_time)/CLOCKS_PER_SEC;
	//cout << "SupervisedModel cost spent: " << duration << " s" << endl;		

	delete[] activations;
	delete[] delta;
	return cost;
}
void ModelCost::gradient(arma::mat& grad){

}
void ModelCost::hessian(arma::mat& grad, arma::mat& hess){

}
void ModelCost::paramsToStack(){

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

		}else if(params[i].params[params_name[ALGORITHM]] == "Pooling"){
			rows_num = ((Pooling*) modules[i])->outputImageNum;
			cols_num = 1;
		}else{
			rows_num = modules[i]->outputSize;
			cols_num = modules[i]->inputSize;
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