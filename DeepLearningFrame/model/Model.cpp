#include "Model.h"
#include "../function/ModelCost.h"
#include "../util/create_optimizer.h"
using namespace dlpft::model;
using namespace dlpft::function;
void Model::pretrain(const arma::mat data, vector<NewParam> params){
	int number_layer = params.size();

	arma::mat features = data;
	if((params[number_layer-1].params[params_name[ALGORITHM]] == "SoftMax")){
		for(int i = 0;i < number_layer-1;i++){
			modules[i]->pretrain(features,params[i]);
			if(i < number_layer-2)
				features = modules[i]->forwardpropagate(features,params[i]);
		}
	}else{
		for(int i = 0;i < number_layer;i++){
			modules[i]->pretrain(features,params[i]);
			if(i < number_layer-1)
			features = modules[i]->forwardpropagate(features,params[i]);
		}
	}
}
void Model::train_classifier(const arma::mat data, const arma::imat labels, vector<NewParam> param){
	arma::mat features = data;
	for(int i = 0;i < layerNumber-1;i++){
		features = modules[i]->forwardpropagate(features,param[i]);
	}
	if(param[layerNumber-1].params[params_name[ALGORITHM]] == "SoftMax"){
		((SoftMax *)modules[layerNumber-1])->train(features,labels,param[layerNumber-1]);
	} 

}
Module* Model::create_module(NewParam& param,int& in_size,int& in_num){
	string m_name = param.params["Algorithm"];
	int out_size = atoi(param.params[params_name[HIDNUM]].c_str());
	string act_func = param.params["Active_function"];
	//cout << act_func << endl;
	ActivationFunction act_choice = get_activation_function(act_func);
	Module* module;
	if(m_name == "AutoEncoder"){
		module = new AutoEncoder(in_size,out_size,act_choice);
		in_size = out_size;
	}else if(m_name == "RBM"){
		module = new RBM(in_size,out_size,act_choice);
		in_size = out_size;
	}else if(m_name == "SC"){
		module = new SparseCoding(in_size,out_size,act_choice);
		in_size = out_size;
	}else if(m_name == "SoftMax"){
		module = new SoftMax(in_size,out_size,act_choice);
		in_size = out_size;
	}else if(m_name == "ConvolveModule"){
		int in_dim = sqrt(in_size / in_num);
		int filter_dim = atoi(param.params[params_name[FILTERDIM]].c_str());
		int out_num = atoi(param.params[params_name[FILTERNUM]].c_str());
		module = new ConvolveModule(in_dim,in_num,filter_dim,out_num,act_choice);
		int out_dim = in_dim - filter_dim + 1;
		in_size = out_dim*out_dim*out_num;
		in_num = out_num;
	}else if(m_name == "CRBM"){
		int in_dim = sqrt(in_size / in_num);
		int filter_dim = atoi(param.params[params_name[FILTERDIM]].c_str());
		int out_num = atoi(param.params[params_name[FILTERNUM]].c_str());
		module = new ConvolutionRBM(in_dim,in_num,filter_dim,out_num,act_choice);
		int out_dim = in_dim - filter_dim + 1;
		in_size = out_dim*out_dim*out_num;
		in_num = out_num;
	}else if(m_name == "Pooling"){
		int in_dim = sqrt(in_size/in_num);
		int pool_dim = atoi(param.params[params_name[POOLINGDIM]].c_str());
		string pool_type = param.params[params_name[POOLINGTYPE]];
		module = new Pooling(in_dim,in_num,pool_dim,pool_type);
		int out_dim = in_dim/pool_dim;
		in_size = out_dim * out_dim * in_num;
		in_num = in_num;
	}else if(m_name == "FullConnection"){
		int o_size = atoi(param.params[params_name[HIDNUM]].c_str());
		module = new FullConnectModule(in_size,o_size,act_choice);
		in_size = o_size;
	}else{
		module = NULL;
	}
	return module;
}
void Model::train(arma::mat data, arma::imat labels,vector<NewParam> model_param){
	int max_epoch = atoi(model_param[0].params[params_name[MAXEPOCH]].c_str());
	int sample_num = data.n_cols;
	int batch_size = atoi(model_param[0].params[params_name[BATCHSIZE]].c_str());
	if (max_epoch == 0)
		max_epoch = 100;
	if(batch_size == 0)
		batch_size = 100;
	arma::mat features = data;
	double error = 0;


	int batch_num = sample_num / batch_size;
	arma::mat *minibatches = new arma::mat[batch_num];


	ModelCost* costfunc = new ModelCost(modules,data,labels,model_param);
	arma::mat grad;

	double learning_rate = atof(model_param[0].params[params_name[LEARNRATE]].c_str());
	if(batch_size == 0) batch_size = 100;
	if(learning_rate == 0) learning_rate = 0.1;
	SgdOptimizer *opt_ptr = new SgdOptimizer(costfunc,max_epoch,learning_rate,batch_size);

	initParams(costfunc->coefficient,model_param);

	opt_ptr->set_func_ptr(costfunc);


	opt_ptr->optimize("SupervisedModel");


	modelParamsToStack(costfunc->coefficient,model_param);

	delete []minibatches;
}
arma::imat Model::predict(const arma::mat testdata, const arma::imat testlabels,vector<NewParam> params){
	int layer_num = params.size();
	arma::mat features = testdata;

	arma::mat max_vals;
	arma::imat pred_labels = zeros<arma::imat>(testdata.n_cols,1);

	for(int i = 0;i < layer_num;i++){
		features = modules[i]->forwardpropagate(features,params[i]);
	}
	if(params[layer_num-1].params["Algorithm"] == "SoftMax"){

		max_vals = max(features);

		for(int i = 0;i < features.n_cols;i++){
			pred_labels[i] = 0;
			for(int j = 0;j < features.n_rows; j++){
				if(max_vals(i) == features(j,i)){
					pred_labels[i] = j+1;
					continue;
				}
			}

		}
	}
	return pred_labels;
}


void Model::initParams(arma::mat& theta,vector<NewParam> param){
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

}
void Model::modelParamsToStack(arma::mat theta,vector<NewParam> params){

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

		}else if(params[i].params[params_name[ALGORITHM]] == "CRBM"){
			int number_filters = ((ConvolutionRBM*) modules[i])->filterNum;
			int filter_dim = ((ConvolutionRBM*) modules[i])->filterDim;
			rows_num = filter_dim*number_filters;
			cols_num = filter_dim;

		}else if(params[i].params[params_name[ALGORITHM]] == "Pooling"){
			rows_num = ((Pooling*) modules[i])->outputImageNum;
			cols_num = 1;
		}else{
			rows_num = modules[i]->outputSize;
			cols_num = modules[i]->inputSize;
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
double Model::predict_acc(const arma::imat predict_labels, const arma::imat labels){
	//fstream ofs;
	//ofs.open("pred_labels.txt",fstream::out);
	//predict_labels.quiet_save(ofs,raw_ascii);
	//ofs.close();
	int sum = 0;
	for(int i = 0;i < predict_labels.size();i++){
		if(predict_labels(i) == labels(i)){
			sum ++;
		}
	}

	return (double)sum/(double)labels.size();
}