#include "FullConnectModule.h"



using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::param;

arma::mat FullConnectModule::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	arma::mat curr_delta = next_layer_weight.t() * next_delta;
	curr_delta = active_function_dev(activeFuncChoice,features) % curr_delta;
	return curr_delta;
}
arma::mat FullConnectModule::forwardpropagate(const arma::mat data,  NewParam param){
	arma::mat features = weightMatrix * data;
	features = active_function(activeFuncChoice,features);
	return features;
}
void FullConnectModule::initial_weights_bias(){
	weightMatrix = 0.005*arma::randu<arma::mat> (outputSize,inputSize);
	bias = zeros(outputSize,1);
}