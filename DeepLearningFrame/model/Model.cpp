#include "Model.h"
#include "../function/ModelCost.h"
#include "../util/create_optimizer.h"
using namespace dlpft::model;
using namespace dlpft::function;
void Model::pretrain(const arma::mat data, vector<NewParam> params){

	// the first layer using data as input
	modules[0]->pretrain(data,params[0]);
	arma::mat features = modules[0]->forwardpropagate(data,params[0]);
	
	if((params[layerNumber-1].params[params_name[ALGORITHM]] == "SoftMax")){
		for(int i = 1;i < layerNumber-1;i++){
			modules[i]->pretrain(features,params[i]);
			if(i < layerNumber-2)
				features = modules[i]->forwardpropagate(features,params[i]);
		}
	}else{
		for(int i = 1;i < layerNumber;i++){
			modules[i]->pretrain(features,params[i]);
			if(i < layerNumber-1)
				features = modules[i]->forwardpropagate(features,params[i]);
		}
	}
}
void Model::train_classifier(const arma::mat data, const arma::mat labels, vector<NewParam> param){
	
	arma::mat features = modules[0]->forwardpropagate(data,param[0]);
	
	for(int i = 1;i < layerNumber-1;i++){
		features = modules[i]->forwardpropagate(features,param[i]);
	}
	if(param[layerNumber-1].params[params_name[ALGORITHM]] == "SoftMax"){
		((SoftMax *)modules[layerNumber-1])->train(features,labels,param[layerNumber-1]);
	}else if(param[layerNumber-1].params[params_name[ALGORITHM]] == "SVM"){
		((SvmModule *)modules[layerNumber-1])->train(features,labels,param[layerNumber-1]);
	}

}
Module* Model::create_module(NewParam& param,int& in_size,int& in_num,int layer_id){
	string m_name = param.params["Algorithm"];
	int out_size = atoi(param.params[params_name[HIDNUM]].c_str());
	string act_func = param.params["Active_function"];
	//cout << act_func << endl;
	string load_w = loadWeightFromFile;
	stringstream layer_id_str;
	layer_id_str << layer_id;
	string w_addr = filePath + "WeightMat_" + layer_id_str.str() + ".txt";
	string b_addr = filePath + "bias_" + layer_id_str.str() + ".txt";
	ActivationFunction act_choice = get_activation_function(act_func);
	double weight_decay = atof(param.params[params_name[WEIGHTDECAY]].c_str());
	Module* module;
	if(m_name == "AutoEncoder"){
		module = new AutoEncoder(in_size,out_size,load_w,w_addr,b_addr,act_choice,weight_decay);
		in_size = out_size;
	}else if(m_name == "RBM"){
		module = new RBM(in_size,out_size,load_w,w_addr,b_addr,act_choice,weight_decay);
		in_size = out_size;
	}else if(m_name == "SC"){
		module = new SparseCoding(in_size,out_size,load_w,w_addr,b_addr,act_choice,weight_decay);
		in_size = out_size;
	}else if(m_name == "SoftMax"){
		module = new SoftMax(in_size,out_size,load_w,w_addr,b_addr,act_choice,weight_decay);
		in_size = out_size;
	}else if(m_name == "SVM"){
		module = new SvmModule();
	}else if(m_name == "ConvolveModule"){
		int in_dim = sqrt(in_size / in_num);
		int filter_dim = atoi(param.params[params_name[FILTERDIM]].c_str());
		int out_num = atoi(param.params[params_name[FEATUREMAPSNUM]].c_str());
		module = new ConvolveModule(in_dim,in_num,filter_dim,out_num,load_w,w_addr,b_addr,act_choice,weight_decay);
		int out_dim = in_dim - filter_dim + 1;
		in_size = out_dim*out_dim*out_num;
		in_num = out_num;
	}else if(m_name == "CRBM"){
		int in_dim = sqrt(in_size / in_num);
		int filter_dim = atoi(param.params[params_name[FILTERDIM]].c_str());
		int out_num = atoi(param.params[params_name[FEATUREMAPSNUM]].c_str());
		module = new ConvolutionRBM(in_dim,in_num,filter_dim,out_num,load_w,w_addr,b_addr,act_choice,weight_decay);
		int out_dim = in_dim - filter_dim + 1;
		in_size = out_dim*out_dim*out_num;
		in_num = out_num;
	}else if(m_name == "Pooling"){
		int in_dim = sqrt(in_size/in_num);
		int pool_dim = atoi(param.params[params_name[POOLINGDIM]].c_str());
		string pool_type = param.params[params_name[POOLINGTYPE]];
		module = new Pooling(in_dim,in_num,pool_dim,pool_type,load_w,w_addr,b_addr);
		int out_dim = in_dim/pool_dim;
		in_size = out_dim * out_dim * in_num;
		in_num = in_num;
	}else if(m_name == "FullConnection"){
		int o_size = atoi(param.params[params_name[HIDNUM]].c_str());
		module = new FullConnectModule(in_size,o_size,load_w,w_addr,b_addr,act_choice,weight_decay);
		in_size = o_size;
	}else{
		module = NULL;
	}
	return module;
}
void Model::train(arma::mat data, arma::mat labels,vector<NewParam> model_param){
	

	cout << "begin train!" <<endl;



	int sample_num = data.n_cols;
	
	double weight_dec = atof(model_param[layerNumber].params[params_name[GLOBALWEIGHTDECAY]].c_str());
	
	
	int batch_size = atoi(model_param[layerNumber].params[params_name[GLOBALBATCHSIZE]].c_str());

	double error = 0;


	int batch_num = sample_num / batch_size;


	ModelCost* costfunc = new ModelCost(modules,data,labels,model_param,weight_dec);
	arma::mat grad;

	//SgdOptimizer *opt_ptr = new SgdOptimizer(costfunc,max_epoch,learning_rate,batch_size,learning_rate_decay);

	Optimizer* opt_ptr = create_optimizer(model_param[layerNumber],costfunc);
	
	
	initParams(costfunc->coefficient,model_param);

	opt_ptr->set_func_ptr(costfunc);


	opt_ptr->optimize("SupervisedModel");


	modelParamsToStack(costfunc->coefficient,model_param);

}
arma::mat Model::predict(const arma::mat testdata, const arma::mat testlabels,vector<NewParam> params){
	arma::mat features = testdata;
	double dropoutfraction = atof(params[layerNumber].params[params_name[DROPOUTFRACTION]].c_str());
	
	arma::mat max_vals;
	arma::mat pred_labels = zeros<arma::mat>(testdata.n_cols,size(testlabels,1));

	for(int i = 0;i < layerNumber-1;i++){
		features = modules[i]->forwardpropagate(features,params[i]);
		if(dropoutfraction > 0){
			features = (1-dropoutfraction)*features;
		}
	}
	if(params[layerNumber-1].params["Algorithm"] == "SoftMax"){
		features = modules[layerNumber-1]->forwardpropagate(features,params[layerNumber-1]);
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
	}else if(params[layerNumber-1].params["Algorithm"] == "SVM"){
		for(int i = 0;i < features.n_cols; i++){
			svm_node* test_case = new svm_node[features.n_rows+1];
			for(int j = 0;j < features.n_rows; j++){
				
				test_case[j].index = j+1;
				test_case[j].value = features(j,i);
			}
			test_case[features.n_rows].index = -1;
			pred_labels[i] = (int)svm_predict(((SvmModule*)modules[layerNumber-1])->svmmodel,test_case);
			delete []test_case;
			test_case = NULL;
		}
	}else{
		features = modules[layerNumber-1]->forwardpropagate(features,params[layerNumber-1]);
		pred_labels = features.t();
	}
	return pred_labels;
}


void Model::initParams(arma::mat& theta,vector<NewParam> param){
	int theta_dim = 0;
	int W_dim = 0;
	int b_dim = 0;
	int curr_loc = 0;
	for(int i = 0;i < layerNumber; i++){
		theta_dim += modules[i]->weightMatrix.size() + modules[i]->bias.size();
	}
	theta = zeros(theta_dim,1);
	for(int i = 0;i < layerNumber; i++){

 		W_dim = modules[i]->weightMatrix.size();

		theta.rows(curr_loc,curr_loc+W_dim-1) = reshape(modules[i]->weightMatrix,W_dim,1);
		curr_loc += W_dim;
		//if(i < layerNumber - 1){
			b_dim = modules[i]->bias.size();
			theta.rows(curr_loc,curr_loc+b_dim-1) = modules[i]->bias;
			curr_loc += b_dim;
		//}
	}

}
void Model::modelParamsToStack(arma::mat theta,vector<NewParam> params){

	int curr_loc = 0;
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
		
		modules[i]->weightMatrix = arma::reshape(theta.rows(curr_loc,curr_loc + modules[i]->weightMatrix.size()-1),rows_num,cols_num);
		curr_loc += modules[i]->weightMatrix.size();
		modules[i]->bias = theta.rows(curr_loc,curr_loc+modules[i]->bias.size()-1);
		curr_loc += modules[i]->bias.size();
	}
}
double Model::predict_acc(const arma::mat predict_labels, arma::mat labels,int class_number){
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