#include "PredictModel.h"
using namespace dlpft::model;
void PredictModel::predict(ResultModel* trainModel,arma::mat& testdata, arma::mat& testlabels,vector<NewParam> params){
	int layer_num = params.size();
	arma::mat features = testdata;
	arma::mat pred_vals;
	arma::mat pred_labels;
	arma::mat max_vals;
	for(int i = 0;i < layer_num;i++){
		if(params[i].params["Algorithm"] == "AutoEncoder"){
			arma::mat activation = trainModel[i].weightMatrix * features  + repmat(trainModel[i].bias,1,features.n_cols);
			features = sigmoid(activation);
		}else if(params[i].params["Algorithm"] == "RBM"){
			arma::mat activation = trainModel[i].weightMatrix * features  + repmat(trainModel[i].bias,1,features.n_cols);
			features = sigmoid(activation);
		}
		else if(params[i].params["Algorithm"] == "SoftMax"){
			pred_vals = trainModel[i].weightMatrix * features;
			max_vals = max(pred_vals);
			pred_labels = zeros(max_vals.size(),1);
			for(int i = 0;i < pred_vals.n_cols;i++){
				for(int j = 0;j < pred_vals.n_rows; j++){
					if(max_vals(i) == pred_vals(j,i)){
						pred_labels[i] = j + 1;
						continue;
					}
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