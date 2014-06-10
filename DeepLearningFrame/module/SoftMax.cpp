#include "SoftMax.h"

#include "../param/SMParam.h"


using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::param;
ResultModel SoftMax::pretrain(const arma::mat data, const arma::imat labels, NewParam param){
	ResultModel result_model;
	typedef Creator<CostFunction> FuncFatory;
	typedef Creator<Optimizer> OptFactory;
	FuncFatory& func_factory = FuncFatory::Instance();
	OptFactory& opt_factory = OptFactory::Instance();

	

	SoftMaxCost* costfunc = new SoftMaxCost(inputSize,outputSize,data,labels);
	
	set_init_coefficient(costfunc->coefficient);

	arma::mat grad;
	Optimizer* testOpt = opt_factory.createProduct(param.params[params_name[OPTIMETHOD]]);



	testOpt->set_func_ptr(costfunc);

	testOpt->optimize("theta");

	result_model.weightMatrix = (costfunc->get_coefficient()).rows(0,outputSize*inputSize-1);
	result_model.weightMatrix.reshape(outputSize,inputSize);
	result_model.algorithm_name = "SoftMax";

	return result_model;

} 
arma::mat SoftMax::backpropagate( ResultModel& result_model,const arma::mat delta,const arma::mat features, const arma::imat labels, NewParam param){
	arma::mat errsum;

	arma::mat curr_delta;
	errsum = result_model.weightMatrix.t() * delta;

	curr_delta = active_function_inv(activeFuncChoice,features) % errsum; 
	return curr_delta;
}
arma::mat SoftMax::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	arma::mat features = result_model.weightMatrix * data;
	features = active_function(activeFuncChoice,features);
	return features;
}
void SoftMax::initial_params(){
	weightMatrix = 0.005*arma::randu<arma::mat> (outputSize,inputSize);
	bias = zeros(outputSize,1);
}
void SoftMax::set_init_coefficient(arma::mat& coefficient){
	coefficient.set_size(outputSize,inputSize);
	coefficient = weightMatrix;
}