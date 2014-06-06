#include "CNN.h"
#include "..\function\AllFunction.h"
#include "..\optimizer\AllOptMethod.h"
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
		last_filter_num = atoi(params[i].params[params_name[FILTERNUM]].c_str());
		last_output_dim = last_output_dim - atoi(params[i].params[params_name[FILTERDIM]].c_str()) + 1;
	}


	typedef Creator<Optimizer> OptFactory;
	OptFactory& opt_factory = OptFactory::Instance();
	CNNCost* costfunc = new CNNCost(modules,data,labels,params);
	arma::mat grad;
	
	Optimizer* testOpt = opt_factory.createProduct(params[0].params[params_name[OPTIMETHOD]]);

	costfunc->set_data(data);
	testOpt->set_func_ptr(costfunc);

	testOpt->optimize("cnn");



	for(int i = 0;i < number_layer; i++){
		delete modules[i];
	}
	delete []modules;
	modules = NULL;
	return resultmodel_ptr;
}