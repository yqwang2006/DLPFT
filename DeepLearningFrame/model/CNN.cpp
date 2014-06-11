#include "CNN.h"
#include "..\function\AllFunction.h"
#include "..\optimizer\AllOptMethod.h"
#include "../module/AllModule.h"d
#include "..\factory\Creator.h"
using namespace dlpft::factory;
using namespace dlpft::model;
ResultModel* CNN::train(const arma::mat data,const arma::imat labels, vector<NewParam> params){
	
	
	
	int max_epoch = atoi(params[0].params[params_name[MAXEPOCH]].c_str());
	int sample_num = data.n_cols;
	int batch_size = atoi(params[0].params[params_name[BATCHSIZE]].c_str());
	ResultModel *resultmodel_ptr = new ResultModel[resultModelSize];
	arma::mat features = data;
	double error = 0;


	int batch_num = sample_num / batch_size;
	arma::mat *minibatches = new arma::mat[batch_num];


	typedef Creator<Optimizer> OptFactory;
	OptFactory& opt_factory = OptFactory::Instance();
	CNNCost* costfunc = new CNNCost(modules,data,labels,params);
	arma::mat grad;
	
	Optimizer* testOpt = opt_factory.createProduct(params[0].params[params_name[OPTIMETHOD]]);

	costfunc->data = data;

	cnnInitParams(costfunc->coefficient,params);

	testOpt->set_func_ptr(costfunc);

	testOpt->optimize("cnn");


	delete []minibatches;
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
void CNN::cnnInitParams(arma::mat& theta,vector<NewParam> param){
	int theta_dim = 0;
	int W_dim = 0;
	int b_dim = 0;
	for(int i = 0;i < layerNumber; i++){
		string algorithm_name = param[i].params[params_name[ALGORITHM]];
		W_dim += ((FullConnectModule *)modules[i])->weightMatrix.size();
		b_dim +=  ((FullConnectModule *)modules[i])->bias.size();

		//last_filter_num = atoi(param[i].params[params_name[FILTERNUM]].c_str());
		//last_output_dim = last_output_dim - atoi(param[i].params[params_name[FILTERDIM]].c_str()) + 1;
	}
	theta_dim = W_dim + b_dim;
	theta.set_size(theta_dim,1);
	arma::mat W = zeros(W_dim,1);
	arma::mat b = zeros(b_dim,1);
	int curr_w_loc = 0;
	int curr_b_loc = 0;
	int next_w_loc = 0;
	int next_b_loc = 0;
	for(int i = 0;i < layerNumber; i++){
		
		arma::mat& weight = modules[i]->weightMatrix;
		next_w_loc += weight.size()-1;
		W.rows(curr_w_loc,next_w_loc) = reshape(weight,weight.size(),1);
		curr_w_loc = next_w_loc;
		arma::mat& bia = modules[i]->bias;
		next_b_loc += bia.size()-1;
		b.rows(curr_b_loc,next_b_loc) = reshape(bia,bia.size(),1);
		curr_b_loc = next_b_loc;
		//last_filter_num = atoi(param[i].params[params_name[FILTERNUM]].c_str());
		//last_output_dim = last_output_dim - atoi(param[i].params[params_name[FILTERDIM]].c_str()) + 1;
	}
	theta.rows(0,W.size()-1) = W;
	theta.rows(W_dim,theta_dim-1) = b;
}