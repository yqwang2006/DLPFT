#include "CNN.h"
#include "..\function\AllFunction.h"
#include "..\optimizer\AllOptMethod.h"
#include "../module/AllModule.h"d
#include "..\factory\Creator.h"
using namespace dlpft::factory;
using namespace dlpft::model;
ResultModel* CNN::train(const arma::mat data,const arma::imat labels, vector<NewParam> params){
	
	
	int number_layer = params.size();
	int max_epoch = atoi(params[0].params[params_name[MAXEPOCH]].c_str());
	int sample_num = data.n_cols;
	int batch_size = atoi(params[0].params[params_name[BATCHSIZE]].c_str());
	ResultModel *resultmodel_ptr = new ResultModel[number_layer];
	arma::mat features = data;
	Module** modules = new Module*[number_layer];
	double error = 0;

	int batch_num = sample_num / batch_size;
	arma::mat *minibatches = new arma::mat[batch_num];

	int last_filter_num = 1;
	int last_output_dim = sqrt(data.n_rows);
	for(int i = 0;i < number_layer;i++){
		modules[i] = create_module(params[i],last_filter_num,last_output_dim);
		if(params[i].params[params_name[ALGORITHM]] == "ConvolveModule"){
			last_filter_num = atoi(params[i].params[params_name[FILTERNUM]].c_str());
			last_output_dim = last_output_dim - atoi(params[i].params[params_name[FILTERDIM]].c_str()) + 1;
	
		}else if(params[i].params[params_name[ALGORITHM]] == "Pooling"){
			last_output_dim = last_output_dim / atoi(params[i].params[params_name[POOLINGDIM]].c_str());
		}
	}


	typedef Creator<Optimizer> OptFactory;
	OptFactory& opt_factory = OptFactory::Instance();
	CNNCost* costfunc = new CNNCost(modules,data,labels,params);
	arma::mat grad;
	
	Optimizer* testOpt = opt_factory.createProduct(params[0].params[params_name[OPTIMETHOD]]);

	costfunc->data = data;
	testOpt->set_func_ptr(costfunc);

	testOpt->optimize("cnn");



	for(int i = 0;i < number_layer; i++){
		delete modules[i];
	}
	delete []modules;
	modules = NULL;
	return resultmodel_ptr;
}
/**
*modules: the cnn net modules,including convolvemodule, pooling, softmax.
*ori_img_dim: the input data dim
*param: the cnn net param. param[i] means the param of the ith module.
*return theta: return weightmat and bias. the seq is:
*theta = [W1,W2,W3,b1,b2,b3];
*pooling module has no weight and bias.
*/
arma::mat CNN::cnnInitParams(Module** modules,int ori_image_size,vector<NewParam> param){
	int theta_dim = 0;
	int num_layer = param.size();
	
	for(int i = 0;i < num_layer; i++){
		string algorithm_name = param[i].params[params_name[ALGORITHM]];
		if(algorithm_name == "FullConnection"){
			
			((FullConnectModule *)modules[i])->initial_params();
		}else if(algorithm_name == "ConvolveModule"){
			
		}else if(algorithm_name == "SoftMax"){
		
		}else{
			//pooling module has no weightmat and bias

		}
		//last_filter_num = atoi(param[i].params[params_name[FILTERNUM]].c_str());
		//last_output_dim = last_output_dim - atoi(param[i].params[params_name[FILTERDIM]].c_str()) + 1;
	}
	arma::mat init;
	return init;
}