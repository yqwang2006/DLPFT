#ifndef ALLPARAM_H
#define ALLPARAM_H
#include "AEParam.h"
#include "SMParam.h"
#include "MultiParam.h"
#include <map>
using namespace dlpft::param;
using namespace std;
namespace dlpft{
	namespace param{
		enum PARAMSNAME { 
			LAYERNUM,LAYERORDER,ALGORITHM,OPTIMETHOD,
			HIDNUM,MAXEPOCH,BATCHSIZE,LEARNRATE,
			POOLINGTYPE,POOLINGDIM,SPARSITY,GAMMA,EPSILON,
			LAMBDA,CLASSESNUM,
			FILTERNUM,FILTERDIM,
			TRAINDATA,TRAINLABELS,TESTDATA,TESTLABELS
		};
		static map<PARAMSNAME,string> params_name;
		static void fill_param_map(){
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
			params_name[GAMMA] = "SC_gamma";
			params_name[EPSILON] = "Epsilon";
			params_name[LAMBDA] = "Lambda";
			params_name[CLASSESNUM] = "Num_classes";
			params_name[TRAINDATA] = "trainData";
			params_name[TRAINLABELS] = "trainLabels";
			params_name[TESTDATA] = "testData";
			params_name[TESTLABELS] = "testLabels";
			params_name[FILTERNUM] = "Filter_num";
			params_name[FILTERDIM] = "Filter_dim";
		}
	};
};
#endif