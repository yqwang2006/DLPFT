#include "PredictModel.h"
using namespace dlpft::model;
void PredictModel::predict(ResultModel* trainModel,arma::mat& testdata, arma::mat& testlabels,vector<NewParam> params){
	int layer_num = params.size();
	arma::mat features = testdata;
	arma::mat pred_vals;
	arma::mat pred_labels;
	arma::mat max_vals;
	Module* single_module;
	for(int i = 0;i < layer_num;i++){
		 single_module = create_module(params[i]);
		
		features = single_module->forwardpropagate(trainModel[i],features,testlabels);
	}
	delete single_module;
	single_module = NULL;
	if(params[layer_num-1].params["Algorithm"] == "SoftMax"){
			max_vals = max(features);
			pred_labels = zeros(max_vals.size(),1);
			for(int i = 0;i < pred_vals.n_cols;i++){
				for(int j = 0;j < pred_vals.n_rows; j++){
					if(max_vals(i) == pred_vals(j,i)){
						if(j == 0)
							pred_labels[i] = features.n_rows;	//number of cases
						else
							pred_labels[i] = j;
						continue;
					}
				}
				
			}
		}
	cout << "Predict error:" << endl;
	cout << 100*(predict_acc(pred_labels,testlabels)) << "%" << endl;
}

double PredictModel::predict_acc(const arma::mat predict_labels, const arma::mat labels){
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