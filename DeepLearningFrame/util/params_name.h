#ifndef PARAMSNAME_H
#define PARAMSNAME_H
#include <map>
using namespace std;
static enum PARAMSNAME { 
	LAYERNUM,LAYERORDER,ALGORITHM,OPTIMETHOD,
	HIDNUM,MAXEPOCH,BATCHSIZE,LEARNRATE,
	SPARSITY,WEIGHTDECAY,KLRHO,
	POOLINGTYPE,POOLINGDIM,EPSILON,
	FEATUREMAPSNUM,FILTERDIM,
	TRAINDATA,TRAINLABELS,TESTDATA,TESTLABELS,FINETUNEDATA,FINETUNELABELS,ACTIVEFUNCTION
};

static map<PARAMSNAME,string> fill_param_map(){
	map<PARAMSNAME,string> params_name;
	params_name[LAYERNUM] = "Layer_num";
	params_name[LAYERORDER] = "Layer_order";
	params_name[ALGORITHM] = "Algorithm";
	params_name[OPTIMETHOD] = "Optimize_method";
	params_name[HIDNUM] = "Hid_num";
	params_name[MAXEPOCH] = "Max_epoch";
	params_name[BATCHSIZE] = "Batch_size";
	params_name[LEARNRATE] = "Learning_rate";
	params_name[POOLINGTYPE] = "Pooling_type";
	params_name[POOLINGDIM] = "Pooling_dim";
	params_name[SPARSITY] = "Sparsity";
	params_name[WEIGHTDECAY] = "Weight_decay";
	params_name[EPSILON] = "Epsilon";
	params_name[KLRHO] = "KL_rho";
	params_name[TRAINDATA] = "trainData";
	params_name[TRAINLABELS] = "trainLabels";
	params_name[TESTDATA] = "testData";
	params_name[TESTLABELS] = "testLabels";
	params_name[FEATUREMAPSNUM] = "Featuremaps_num";
	params_name[FILTERDIM] = "Filter_dim";
	params_name[FINETUNEDATA]="Finetune_data";
	params_name[FINETUNELABELS] = "Finetune_labels";
	params_name[ACTIVEFUNCTION]="Active_function";
	return params_name;
}
static map<PARAMSNAME,string> params_name = fill_param_map();
#endif