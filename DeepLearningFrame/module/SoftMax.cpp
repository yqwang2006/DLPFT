#include "SoftMax.h"

#include "../util/create_optimizer.h"

using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::param;
void SoftMax::train(const arma::mat data, const arma::mat labels, NewParam param){

	typedef Creator<CostFunction> FuncFatory;
	typedef Creator<Optimizer> OptFactory;
	FuncFatory& func_factory = FuncFatory::Instance();
	OptFactory& opt_factory = OptFactory::Instance();

	

	SoftMaxCost* costfunc = new SoftMaxCost(inputSize,outputSize,data,labels);

	set_init_coefficient(costfunc->coefficient);

	arma::mat grad;
	Optimizer* testOpt = create_optimizer(param,costfunc);
	testOpt->optimize("theta");

	weightMatrix = costfunc->coefficient.rows(0,outputSize*inputSize-1);
	weightMatrix.reshape(outputSize,inputSize);

	 bias = costfunc->coefficient.rows(outputSize*inputSize,costfunc->coefficient.size()-1);


} 
arma::mat SoftMax::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	arma::mat curr_delta = next_delta;
	return curr_delta;
}
arma::mat SoftMax::forwardpropagate(const arma::mat data,  NewParam param){

	arma::mat features = weightMatrix * data + repmat(bias,1,data.n_cols);
	//arma::mat features = weightMatrix * data;
	features = active_function(SOFTMAX,features);
	return features;

}
void SoftMax::initial_weights_bias(){
	if(load_weight == "YES"){
		if(weight_addr != "" && bias_addr != ""){
			if(initial_weights_bias_from_file(weight_addr,bias_addr)){
				return;
			}
		}
	}
		srand(unsigned(time(NULL)));
		double r = sqrt(6) / sqrt(outputSize + inputSize + 1);
		int h_v_size = outputSize * inputSize;
	#if DEBUG
		weightMatrix = 0.13*arma::ones<arma::mat> (outputSize,inputSize);
	#else
		weightMatrix = arma::randu<arma::mat> (outputSize,inputSize)*2*r-r;
	#endif
		weightMatrix = 0.01 * (arma::randu<arma::mat> (outputSize,inputSize) - 0.5);


		bias = arma::zeros(outputSize,1);
	
}
void SoftMax::set_init_coefficient(arma::mat& coefficient){
	coefficient.set_size(outputSize*inputSize + outputSize,1);
	coefficient.rows(0,outputSize*inputSize-1) = reshape(weightMatrix,weightMatrix.size(),1);
	coefficient.rows(outputSize*inputSize,coefficient.size()-1) = bias;
	/*coefficient.set_size(outputSize*inputSize,1);
	coefficient.rows(0,outputSize*inputSize-1) = reshape(weightMatrix,weightMatrix.size(),1);*/
}
void SoftMax::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,double weight_decay, arma::mat& Wgrad, arma::mat& bgrad){

	Wgrad = ((double)1/input_data.n_cols)*delta * input_data.t() + weight_decay * weightMatrix;
	bgrad = sum(delta,1)/input_data.n_cols;
}