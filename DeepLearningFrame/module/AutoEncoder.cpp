#include "AutoEncoder.h"

using namespace dlpft::param;
using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
void AutoEncoder::pretrain(const arma::mat data, const arma::imat labels, NewParam param){
	
	typedef Creator<Optimizer> OptFactory;
	OptFactory& opt_factory = OptFactory::Instance();

	int h_v_size = inputSize * outputSize;
	int sample_num = data.n_cols;

	SAECostFunction* costfunc = new SAECostFunction(inputSize,outputSize);
	arma::mat grad;
	Optimizer* testOpt = opt_factory.createProduct(param.params[params_name[OPTIMETHOD]]);

	costfunc->data = data;
	costfunc->labels = labels;
	set_init_coefficient(costfunc->coefficient);
	
	testOpt->set_func_ptr(costfunc);


	//cout << "before opt:" << costfunc->get_coefficient()->n_rows<<";" << costfunc->get_coefficient()->n_cols << endl;
	//cout << ((AEParam*)param)->get_max_epoch() << endl;
	testOpt->set_max_iteration(atoi(param.params[params_name[MAXEPOCH]].c_str()));
	testOpt->optimize("theta"); 

	//cout << "after opt:" << costfunc->get_coefficient()->n_rows<<";" << costfunc->get_coefficient()->n_cols << endl;
	
	
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
void AutoEncoder::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad){
	int lambda = atoi(param.params[params_name[LAMBDA]].c_str());
	Wgrad = ((double)1/input_data.n_cols)*delta * input_data.t() + lambda * weightMatrix;
	bgrad = sum(delta,1)/input_data.n_cols;
}