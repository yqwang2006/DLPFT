#include "SoftMax.h"

#include "../param/SMParam.h"


using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::param;
ResultModel SoftMax::pretrain(const arma::mat data, const arma::mat labels, NewParam param){
	ResultModel result_model;
	typedef Creator<CostFunction> FuncFatory;
	typedef Creator<Optimizer> OptFactory;
	FuncFatory& func_factory = FuncFatory::Instance();
	OptFactory& opt_factory = OptFactory::Instance();

	
	int visiable_size = data.n_rows;
	int n_classes = atoi(param.params["Num_classes"].c_str());

	SoftMaxCost* costfunc = new SoftMaxCost(visiable_size,n_classes,data,labels);
	arma::mat grad;
	Optimizer* testOpt = opt_factory.createProduct(param.params["Optimize_method"]);


	testOpt->set_func_ptr(costfunc);

	testOpt->optimize("theta");

	result_model.weightMatrix = (costfunc->get_coefficient()).rows(0,n_classes*visiable_size-1);
	result_model.weightMatrix.reshape(n_classes,visiable_size);
	result_model.algorithm_name = "SoftMax";

	return result_model;

} 
arma::mat SoftMax::backpropagate( ResultModel& result_model,const arma::mat delta,const arma::mat features, const arma::mat labels, NewParam param){
	double errsum = 0;

	arma::mat desired_out = zeros(features.n_rows,features.n_cols);
	for(int i = 0;i < features.n_cols; i++){
		if(labels(i) == features.n_rows)
			desired_out(0,i) = 1;
		else
			desired_out(labels(i),i) = 1;
	}

	arma::mat curr_delta;
	errsum = sum(sum(result_model.weightMatrix.t() * delta));

	curr_delta = features * (1-features) * errsum; 
	return curr_delta;
}
arma::mat SoftMax::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::mat labels){
	arma::mat features = result_model.weightMatrix * data;
	features = sigmoid(features);
	return features;
}