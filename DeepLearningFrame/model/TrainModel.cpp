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
	return resultmodel_ptr;
}
Module* TrainModel::create_module(NewParam& param){
	string m_name = param.params["Algorithm"];

	Module* module;
	if(m_name == "AutoEncoder"){
		module = new AutoEncoder();
	}else if(m_name == "RBM"){
		module = new RBM();
	}else if(m_name == "SC"){
		module = new SparseCoding();
	}else if(m_name == "SoftMax"){
		module = new SoftMax();
	}
	return module;
}