#include "CNN.h"
#include "..\function\AllFunction.h"
#include "..\optimizer\AllOptMethod.h"
#include "../module/AllModule.h"d
#include "..\factory\Creator.h"
using namespace dlpft::factory;
using namespace dlpft::model;
void CNN::train(const arma::mat data,const arma::imat labels, vector<NewParam> params){
	
	
	
	int max_epoch = atoi(params[0].params[params_name[MAXEPOCH]].c_str());
	int sample_num = data.n_cols;
	int batch_size = atoi(params[0].params[params_name[BATCHSIZE]].c_str());
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


	cnnParamsToStack(costfunc->coefficient,params);

	delete []minibatches;
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
		
		W_dim += modules[i]->weightMatrix.size();
		b_dim += modules[i]->bias.size();

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
		next_w_loc = curr_w_loc + weight.size();
		W.rows(curr_w_loc,next_w_loc-1) = reshape(weight,weight.size(),1);
		curr_w_loc = next_w_loc;
		arma::mat& bia = modules[i]->bias;
		next_b_loc = curr_b_loc + bia.size();
		b.rows(curr_b_loc,next_b_loc-1) = reshape(bia,bia.size(),1);
		curr_b_loc = next_b_loc;
		//last_filter_num = atoi(param[i].params[params_name[FILTERNUM]].c_str());
		//last_output_dim = last_output_dim - atoi(param[i].params[params_name[FILTERDIM]].c_str()) + 1;
	}
	theta.rows(0,W.size()-1) = W;
	theta.rows(W_dim,theta_dim-1) = b;
//#if DEBUG
//	ofstream ofs;
//	ofs.open("theta.txt");
//	theta.quiet_save(ofs,raw_ascii);
//	ofs.close();
//	ofs.open("reshape.txt");
//	arma::mat weight = reshape(theta.rows(0,1619),9*20,9);
//	weight.quiet_save(ofs,raw_ascii);
//	ofs.close();
//#endif

}
void CNN::predict(arma::mat& testdata, arma::imat& testlabels,vector<NewParam> params){
	int layer_num = params.size();
	arma::mat features = testdata;

	arma::mat max_vals;
	for(int i = 0;i < layer_num;i++){
		features = modules[i]->forwardpropagate(features,params[i]);
	}
	if(params[layer_num-1].params["Algorithm"] == "SoftMax"){

		max_vals = max(features);
		arma::imat pred_labels(max_vals.size(),1);
		for(int i = 0;i < features.n_cols;i++){
			pred_labels[i] = 0;
			for(int j = 0;j < features.n_rows; j++){
				if(max_vals(i) == features(j,i)){
					if(j == 0)
						pred_labels[i] = features.n_rows;	//number of cases
					else
						pred_labels[i] = j;
					continue;
				}
			}

		}
		cout << "Predict accu:" << endl;
		cout << 100*(predict_acc(pred_labels,testlabels)) << "%" << endl;
	}

}

double CNN::predict_acc(const arma::imat predict_labels, const arma::imat labels){
	fstream ofs;
	ofs.open("pred_labels.txt",fstream::out);
	predict_labels.quiet_save(ofs,raw_ascii);
	ofs.close();
	int sum = 0;
	for(int i = 0;i < predict_labels.size();i++){
		if(predict_labels(i) == labels(i)){
			sum ++;
		}
	}

	return (double)sum/(double)labels.size();
}
void CNN::cnnParamsToStack(arma::mat theta,vector<NewParam> params){

	int start_w_loc = 0;
	int start_b_loc = 0;
	int end_w_loc = 0;
	int end_b_loc = 0;
	for(int i = 0;i < layerNumber; i++){
		int hiddenSize = 0;
		int rows_num = 0,cols_num = 0;
		if(params[i].params[params_name[ALGORITHM]] == "ConvolveModule"){
			int number_filters = ((ConvolveModule*) modules[i])->filterNum;
			int filter_dim = ((ConvolveModule*) modules[i])->filterDim;
			rows_num = filter_dim*number_filters;
			cols_num = filter_dim;

		}else if(params[i].params[params_name[ALGORITHM]] == "FullConnection"){
			rows_num = ((FullConnectModule*) modules[i])->outputSize;
			cols_num = ((FullConnectModule*) modules[i])->inputSize;
		}else if(params[i].params[params_name[ALGORITHM]] == "SoftMax"){
			rows_num = ((SoftMax*) modules[i])->outputSize;
			cols_num = ((SoftMax*) modules[i])->inputSize;

		}else if(params[i].params[params_name[ALGORITHM]] == "Pooling"){
			rows_num = ((Pooling*) modules[i])->outputImageNum;
			cols_num = 1;
		}
		end_w_loc += modules[i]->weightMatrix.size();
		modules[i]->weightMatrix = arma::reshape(theta.rows(start_w_loc,end_w_loc-1),rows_num,cols_num);
		start_w_loc = end_w_loc;
	}
	start_b_loc = end_w_loc;
	end_b_loc = end_w_loc;
	for(int i = 0;i < layerNumber; i++){
		int hiddenSize = 0;

		end_b_loc += modules[i]->bias.size();
		arma::mat b = arma::reshape(theta.rows(start_b_loc,end_b_loc-1),end_b_loc-start_b_loc,1);
		modules[i]->bias = b;

		start_b_loc = end_b_loc;
	}
}