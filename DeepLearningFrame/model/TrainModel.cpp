#include "TrainModel.h"
using namespace dlpft::factory;
using namespace dlpft::model;
ResultModel* TrainModel::pretrain(arma::mat& data,arma::mat& labels, vector<NewParam> params){
	int number_layer = params.size();

	ResultModel *resultmodel_ptr = new ResultModel[number_layer];
	arma::mat features = data;
	Module* single_module;

	for(int i = 0;i < number_layer;i++){
		single_module = create_module(params[i]);
		resultmodel_ptr[i] = single_module->pretrain(features,labels,params[i]);
		features = single_module->forwardpropagate(resultmodel_ptr[i],features,labels);
	}
	delete single_module;
	single_module = NULL;
	return resultmodel_ptr;
}
