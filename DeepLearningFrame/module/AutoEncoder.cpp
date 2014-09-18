#include "AutoEncoder.h"
#include "../util/create_optimizer.h"
using namespace dlpft::param;
using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
void AutoEncoder::pretrain(const arma::mat data, NewParam param){
	

	int h_v_size = inputSize * outputSize;
	int sample_num = data.n_cols;
	double sparsity_coeff = atof(param.params[params_name[SPARSITY]].c_str());
	double weight_decay_rate = atof(param.params[params_name[WEIGHTDECAY]].c_str());
	double KL_Rho_dist = atof(param.params[params_name[KLRHO]].c_str());

	SAECostFunction* costfunc = new SAECostFunction(inputSize,outputSize,sparsity_coeff,weight_decay_rate,KL_Rho_dist);
	arma::mat grad;
	costfunc->data = data;
	costfunc->labels = zeros<arma::mat>(data.n_cols,1);
	set_init_coefficient(costfunc->coefficient);
	
	Optimizer* testOpt = create_optimizer(param,costfunc);
	testOpt->set_func_ptr(costfunc);
	testOpt->optimize("theta"); 

	
	
	weightMatrix = reshape((costfunc->coefficient).rows(0,h_v_size-1),outputSize,inputSize);
	bias = (costfunc->coefficient).rows(2*h_v_size,2*h_v_size+outputSize-1);

}

arma::mat AutoEncoder::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	arma::mat curr_delta = active_function_dev(activeFuncChoice,features) % next_delta; 
	return curr_delta;
}
arma::mat AutoEncoder::forwardpropagate(const arma::mat data,  NewParam param){
	//weightMat: hidden_size * visible_size
	//bias: (hidden_size,1)
	arma::mat activation = weightMatrix * data + repmat(bias,1,data.n_cols);
	activation = active_function(activeFuncChoice,activation);
	return activation;
}
void AutoEncoder::initial_weights_bias(){
	
	if(load_weight == "YES"){
		if(weight_addr != "" && bias_addr != ""){
			if(initial_weights_bias_from_file(weight_addr,bias_addr)){
				backwardWeight = weightMatrix.t();
				backwardBias = arma::zeros(inputSize,1);
				//backwardWeight = arma::randu<arma::mat> (inputSize,outputSize)*2*r-r;
				return;
			}
		}
	}
		srand(unsigned(time(NULL)));
		double r = sqrt(6) / sqrt(outputSize + inputSize + 1);
		int h_v_size = outputSize * inputSize;
	
		weightMatrix = arma::randu<arma::mat> (outputSize,inputSize)*2*r-r;
		backwardWeight = arma::randu<arma::mat> (inputSize,outputSize)*2*r-r;
		bias = arma::zeros(outputSize,1);
		backwardBias = arma::zeros(inputSize,1);
	
}
void AutoEncoder::set_init_coefficient(arma::mat& coefficient){
	coefficient.set_size(outputSize * inputSize * 2 + outputSize + inputSize);
	int h_v_size = inputSize * outputSize;
	coefficient.rows(0,h_v_size-1) = reshape(weightMatrix,h_v_size,1);
	coefficient.rows(h_v_size,2*h_v_size-1) = reshape(backwardWeight,h_v_size,1);
	coefficient.rows(2*h_v_size,2 * h_v_size+outputSize-1) = bias;
	coefficient.rows(2*h_v_size+outputSize,coefficient.size()-1) = backwardBias;
}
void AutoEncoder::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, double weight_decay, arma::mat& Wgrad, arma::mat& bgrad){
	
	Wgrad = ((double)1/input_data.n_cols)*delta * input_data.t();// + weightDecay * weightMatrix;
	bgrad = sum(delta,1)/input_data.n_cols;
}