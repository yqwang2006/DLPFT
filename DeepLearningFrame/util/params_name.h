#ifndef PARAMSNAME_H
#define PARAMSNAME_H
#include <map>
using namespace std;
static enum PARAMSNAME { 
	MODELTYPE,GLOBALOPTMETHOD,GLOBALMAXEPOCH,GLOBALBATCHSIZE,GLOBALWEIGHTDECAY,GLOBALLEARNRATE,GLOBALLEARNRATEDECAY,
	LAYERNUM,LAYERORDER,ALGORITHM,OPTIMETHOD,
	HIDNUM,MAXEPOCH,BATCHSIZE,LEARNRATE,MOMENTUM,
	SPARSITY,WEIGHTDECAY,KLRHO,
	POOLINGTYPE,POOLINGDIM,EPSILON,
	FINETUNESWITCH,FEATUREMAPSNUM,FILTERDIM,
	TRAINDATA,TRAINLABELS,TESTDATA,TESTLABELS,FINETUNEDATA,FINETUNELABELS,ACTIVEFUNCTION,
	LOADWEIGHT,WEIGHTADDRESS,BIASADDRESS,SVMTYPE,SVMKERNELTYPE,SVMEPSILON,SVMCOST,DROPOUTFRACTION,INPUTMASKEDZEROFRACTION,
	ELMTYPE, ELM_C, ELMCLASSNUM
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
	params_name[FEATUREMAPSNUM] = "Features_num";
	params_name[FILTERDIM] = "Filter_dim";
	params_name[FINETUNEDATA]="Finetune_data";
	params_name[FINETUNELABELS] = "Finetune_labels";
	params_name[ACTIVEFUNCTION]="Active_function";
	params_name[MODELTYPE] = "Model_type";
	params_name[FINETUNESWITCH] = "Finetune_switch";
	params_name[GLOBALOPTMETHOD] = "Global_optimize_method";
	params_name[GLOBALMAXEPOCH] = "Global_max_epoch";
	params_name[GLOBALBATCHSIZE] = "Global_batch_size";
	params_name[GLOBALLEARNRATE] = "Global_learning_rate";
	params_name[GLOBALLEARNRATEDECAY] = "Global_learning_rate_decayrate";
	params_name[LOADWEIGHT] = "Init_weight_from_file";
	params_name[WEIGHTADDRESS] = "Weight_addr";
	params_name[BIASADDRESS] = "Bias_addr";
	params_name[SVMTYPE] = "Svm_type";
	params_name[SVMKERNELTYPE] = "Svm_kernel_type";
	params_name[SVMEPSILON] = "Svm_epsilon";
	params_name[SVMCOST] = "Svm_cost";
	params_name[DROPOUTFRACTION] = "Dropout_fraction";
	params_name[INPUTMASKEDZEROFRACTION] = "InputMaskedZeroFraction";
	params_name[MOMENTUM] = "Momentum";
	params_name[ELMTYPE] = "Elm_type";
	params_name[ELM_C] = "Elm_C";
	params_name[ELMCLASSNUM] = "Elm_classnumber";
	return params_name;
}
static map<PARAMSNAME,string> params_name = fill_param_map();
#endif