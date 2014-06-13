#include "FullConnectModule.h"



using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::param;

arma::mat FullConnectModule::forwardpropagate(const arma::mat data,  NewParam param){
	arma::mat features = weightMatrix * data + repmat(bias,1,data.n_cols);
	features = active_function(activeFuncChoice,features);
	return features;
}
void FullConnectModule::initial_weights_bias(){
	weightMatrix = 0.005*arma::randu<arma::mat> (outputSize,inputSize);
	bias = zeros(outputSize,1);
}
arma::mat FullConnectModule::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	arma::mat curr_delta = active_function_dev(activeFuncChoice,features) % next_delta; 
	return curr_delta;
}
void FullConnectModule::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad){
	int lambda = atoi(param.params[params_name[LAMBDA]].c_str());
	Wgrad = ((double)1/input_data.n_cols)*delta * input_data.t() + lambda * weightMatrix;
	bgrad = sum(delta,1)/input_data.n_cols;
}