#include "AutoEncoder.h"

using namespace dlpft::param;
using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
ResultModel AutoEncoder::pretrain(const arma::mat data, const arma::imat labels, NewParam param){
	
	ResultModel result_model;

	typedef Creator<Optimizer> OptFactory;
	OptFactory& opt_factory = OptFactory::Instance();

	int h_v_size = inputSize * outputSize;
	int sample_num = data.n_cols;

	SAECostFunction* costfunc = new SAECostFunction(inputSize,outputSize);
	arma::mat grad;
	Optimizer* testOpt = opt_factory.createProduct(param.params[params_name[OPTIMETHOD]]);

	costfunc->data = data;

	set_init_coefficient(costfunc->coefficient);
	
	testOpt->set_func_ptr(costfunc);


	//cout << "before opt:" << costfunc->get_coefficient()->n_rows<<";" << costfunc->get_coefficient()->n_cols << endl;
	//cout << ((AEParam*)param)->get_max_epoch() << endl;
	testOpt->set_max_iteration(atoi(param.params[params_name[MAXEPOCH]].c_str()));
	testOpt->optimize("theta"); 

	//cout << "after opt:" << costfunc->get_coefficient()->n_rows<<";" << costfunc->get_coefficient()->n_cols << endl;
	
	result_model.algorithm_name = "AutoEncoder";
	result_model.weightMatrix = (costfunc->coefficient).rows(0,h_v_size-1);
	result_model.weightMatrix.reshape(outputSize,inputSize);
	result_model.bias = (costfunc->coefficient).rows(2*h_v_size,2*h_v_size+outputSize-1);

	
	return result_model;
}

arma::mat AutoEncoder::backpropagate( ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param){
	arma::mat curr_delta;


	return curr_delta;

}
arma::mat AutoEncoder::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	arma::mat activation = result_model.weightMatrix * data + repmat(result_model.bias,1,data.n_cols);
	activation = active_function(activeFuncChoice,activation);
	return activation;
}
void AutoEncoder::initial_params(){
	
	double r = sqrt(6) / sqrt(outputSize + inputSize + 1);
	int h_v_size = outputSize * inputSize;
	
	forwardWeight = arma::randu<arma::mat> (outputSize,inputSize)*2*r-r;
	backwardWeight = arma::randu<arma::mat> (inputSize,outputSize)*2*r-r;
	forwardBias = arma::zeros(outputSize,1);
	backwardBias = arma::zeros(inputSize,1);
}
void AutoEncoder::set_init_coefficient(arma::mat& coefficient){
	coefficient.set_size(outputSize * inputSize * 2 + outputSize + inputSize);
	int h_v_size = inputSize * outputSize;
	forwardWeight.reshape(h_v_size,1);
	backwardWeight.reshape(h_v_size,1);
	/*arma::mat W1 = arma::ones(hiddenSize,visiableSize)*2*r-r;
	arma::mat W2 = arma::ones(visiableSize,hiddenSize)*r-r;
	W1.reshape(h_v_size,1);
	W2.reshape(h_v_size,1);*/
	coefficient.rows(0,h_v_size-1) = forwardWeight;
	coefficient.rows(h_v_size,2*h_v_size-1) = backwardWeight;
	coefficient.rows(2*h_v_size,2 * h_v_size+outputSize-1) = forwardBias;
	coefficient.rows(2*h_v_size+outputSize,coefficient.size()-1) = backwardBias;
}