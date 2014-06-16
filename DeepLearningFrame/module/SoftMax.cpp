#include "SoftMax.h"

#include "../param/SMParam.h"


using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::param;
void SoftMax::pretrain(const arma::mat data, const arma::imat labels, NewParam param){

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

	weightMatrix = costfunc->coefficient.rows(0,outputSize*inputSize-1);
	weightMatrix.reshape(outputSize,inputSize);

	 bias = costfunc->coefficient.rows(outputSize*inputSize,costfunc->coefficient.size()-1);


} 
arma::mat SoftMax::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
//#if DEBUG
	arma::mat curr_delta = next_delta;
//#else
//	arma::mat curr_delta = active_function_dev(activeFuncChoice,features) % next_delta;
//#endif
	return curr_delta;
}
arma::mat SoftMax::forwardpropagate(const arma::mat data,  NewParam param){

	arma::mat features = weightMatrix * data + repmat(bias,1,data.n_cols);
	//arma::mat features = weightMatrix * data;
	features = active_function(activeFuncChoice,features);
	return features;

}
void SoftMax::initial_weights_bias(){
	srand(unsigned(time(NULL)));
	double r = sqrt(6) / sqrt(outputSize + inputSize + 1);
	int h_v_size = outputSize * inputSize;
#if DEBUG
	weightMatrix = 0.13*arma::ones<arma::mat> (outputSize,inputSize);
#else
	weightMatrix = arma::randu<arma::mat> (outputSize,inputSize)*2*r-r;
#endif
	
	bias = arma::zeros(outputSize,1);
}
void SoftMax::set_init_coefficient(arma::mat& coefficient){
	coefficient.set_size(outputSize*inputSize + outputSize,1);
	coefficient.rows(0,outputSize*inputSize-1) = reshape(weightMatrix,weightMatrix.size(),1);
	coefficient.rows(outputSize*inputSize,coefficient.size()-1) = bias;
	/*coefficient.set_size(outputSize*inputSize,1);
	coefficient.rows(0,outputSize*inputSize-1) = reshape(weightMatrix,weightMatrix.size(),1);*/
}
void SoftMax::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad){
	int lambda = atoi(param.params[params_name[LAMBDA]].c_str());
	Wgrad = ((double)1/input_data.n_cols)*delta * input_data.t() + 3e-3 * weightMatrix;
	bgrad = sum(delta,1)/input_data.n_cols;
}