#include "FullConnectModule.h"



using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::param;

arma::mat FullConnectModule::backpropagate( ResultModel& result_model,const arma::mat delta,const arma::mat features, const arma::imat labels, NewParam param){
	arma::mat errsum;

	arma::mat curr_delta;
	errsum = result_model.weightMatrix.t() * delta;

	curr_delta = active_function_inv(active_func_choice,features) % errsum; 
	return curr_delta;
}
arma::mat FullConnectModule::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	arma::mat features = result_model.weightMatrix * data;
	features = active_function(active_func_choice,features);
	return features;
}