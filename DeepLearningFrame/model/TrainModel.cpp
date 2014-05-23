#include "TrainModel.h"
using namespace dlpft::factory;
using namespace dlpft::model;
ResultModel* TrainModel::train(arma::mat& data,arma::mat& labels, vector<NewParam> params){
	int number_layer = params.size();
	
	ResultModel *resultmodel_ptr = new ResultModel[number_layer];
	
	
	
	if(number_layer >= 2){
		resultmodel_ptr[0] = single_layer_train(data,labels,params[0]);

		for(int i = 1;i < number_layer;i++){
			resultmodel_ptr[i] = single_layer_train(resultmodel_ptr[i-1].features,labels,params[i]);
		}

	}else{
		resultmodel_ptr[0] = single_layer_train(data,labels,params[0]);
	}
	return resultmodel_ptr;
}
ResultModel TrainModel::single_layer_train(arma::mat& data,arma::mat& labels, NewParam& param){
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

	ResultModel rm = module->run(data,labels,param);
	delete module;
	return rm;
}