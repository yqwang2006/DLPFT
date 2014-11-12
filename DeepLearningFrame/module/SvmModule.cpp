#include "SvmModule.h"


#include "../util/create_optimizer.h"


using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::param;
void SvmModule::train(const arma::mat data, const arma::mat labels, NewParam param){

	
	int num_featurtes = data.n_rows;
	int num_samples = data.n_cols;
	
	svm_parameter svm_param;

	prob.l = num_samples;

	double* y_space = new double[num_samples];
	svm_node** x_space = new svm_node*[num_samples];
	for(int i = 0;i < num_samples; i++){
		x_space[i] = new svm_node[num_featurtes+1];
	}

	for(int i = 0;i < num_samples; i++){
		y_space[i] = (double)labels(i);
		for(int j = 0; j < num_featurtes; j++){
			x_space[i][j].index = j+1;
			x_space[i][j].value = data(j,i);
		
		}
		x_space[i][num_featurtes].index = -1;
	}

	prob.x = x_space;
	prob.y = y_space;

	svm_param.svm_type = atoi(param.params[params_name[SVMTYPE]].c_str());
	svm_param.kernel_type = atoi(param.params[params_name[SVMKERNELTYPE]].c_str());
	svm_param.eps = atof(param.params[params_name[SVMEPSILON]].c_str());
	svm_param.C = atof(param.params[params_name[SVMCOST]].c_str());
	svm_param.degree = 3;
	svm_param.gamma = 0.03125;             //RBF重要参数：g
	svm_param.coef0 = 0;
	svm_param.nu = 0.5;
	svm_param.cache_size = 100;             //重要参数：c (松弛变量)
	svm_param.p = 0.1;
	svm_param.shrinking = 1;
	svm_param.probability = 0;
	svm_param.nr_weight = 0;
	svm_param.weight_label = NULL;
	svm_param.weight = NULL;

	svmmodel = svm_train(&prob, &svm_param);
	svm_save_model("model.txt",svmmodel);

} 
arma::mat SvmModule::backpropagate(const arma::mat next_delta, const arma::mat features, NewParam param){
	arma::mat curr_delta = next_delta;
	return curr_delta;
}
arma::mat SvmModule::forwardpropagate(const arma::mat data,  NewParam param){

	arma::mat features;
	return features;

}
void SvmModule::initial_weights_bias(){
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
void SvmModule::set_init_coefficient(arma::mat& coefficient){
	coefficient.set_size(outputSize*inputSize + outputSize,1);
	coefficient.rows(0,outputSize*inputSize-1) = reshape(weightMatrix,weightMatrix.size(),1);
	coefficient.rows(outputSize*inputSize,coefficient.size()-1) = bias;
	/*coefficient.set_size(outputSize*inputSize,1);
	coefficient.rows(0,outputSize*inputSize-1) = reshape(weightMatrix,weightMatrix.size(),1);*/
}
void SvmModule::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,double weight_decay, arma::mat& Wgrad, arma::mat& bgrad){

	Wgrad = ((double)1/input_data.n_cols)*delta * input_data.t() + weight_decay * weightMatrix;
	bgrad = sum(delta,1)/input_data.n_cols;
}