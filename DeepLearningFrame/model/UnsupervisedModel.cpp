#include "UnsupervisedModel.h"
#include "../util/onehot.h"
using namespace dlpft::factory;
using namespace dlpft::model;
void UnsupervisedModel::pretrain(const arma::mat data,const arma::imat labels, vector<NewParam> params){
	int number_layer = params.size();

	arma::mat features = data;

	for(int i = 0;i < number_layer;i++){
		modules[i]->pretrain(features,labels,params[i]);
		features = modules[i]->forwardpropagate(features,params[i]);
	}

	if(finetune_switch){
		assert(params[number_layer-1].params[params_name[ALGORITHM]] == "SoftMax");
		finetune_BP(data,labels,params);
	}
}
void UnsupervisedModel::finetune_BP(const arma::mat data, const arma::imat labels, vector<NewParam> params){
	int number_layer = params.size();
	int samples_num = data.n_cols;
	arma::mat* features = new arma::mat[number_layer];
	arma::mat* delta = new arma::mat[number_layer+1];
	arma::mat* Wgrad = new arma::mat[number_layer];
	arma::mat* bgrad = new arma::mat[number_layer];

	int max_epoch = 5;

	int batch_size = 100;

	arma::mat minibatch;
	arma::imat minibatch_labels;
	int upper_bound = 0;
	int batch_number = samples_num / batch_size;

	for(int epoch = 0; epoch < max_epoch; epoch ++){
		for(int n = 0;n < batch_number; n++){
			if((n+1)*batch_size-1 > samples_num-1)
				upper_bound = samples_num - 1;
			else
				upper_bound = (n+1)*batch_size-1;
			minibatch = data.cols(n*batch_size,upper_bound);
			minibatch_labels = labels.rows(n*batch_size,upper_bound);
			for(int i = 0;i < number_layer ;i++){
				if(i == 0){
					features[i] = modules[i]->forwardpropagate(minibatch,params[i]);
				}else{
					features[i] = modules[i]->forwardpropagate(features[i-1],params[i]);
				}
			}

			arma::mat desired_out = onehot(features[number_layer-1].n_rows,features[number_layer-1].n_cols,minibatch_labels);

			// compute the output delta
			delta[number_layer] = (desired_out - features[number_layer-1]);
			arma::mat next_layer_weight;
			arma::mat next_delta;
			for(int i = number_layer-1;i >=0 ;i--){
				
				if(i == number_layer-1){
					delta[i] = modules[i]->backpropagate(next_layer_weight,delta[i+1],features[i],params[i]);
					modules[i]->calculate_grad_using_delta(features[i-1],delta[i],params[i],Wgrad[i],bgrad[i]);
				}else if(i == 0){
					delta[i] = modules[i]->backpropagate(modules[i+1]->weightMatrix,next_delta,features[i],params[i]);
					modules[i]->calculate_grad_using_delta(minibatch,delta[i],params[i],Wgrad[i],bgrad[i]);
				}else{
					delta[i] = modules[i]->backpropagate(modules[i+1]->weightMatrix,next_delta,features[i],params[i]);
					modules[i]->calculate_grad_using_delta(features[i-1],delta[i],params[i],Wgrad[i],bgrad[i]);
				}
				next_delta = modules[i]->process_delta(delta[i]);
			}

			for(int i = 0;i < number_layer; i++){
				double learn_rate = atof(params[i].params["Learning_rate"].c_str());

				modules[i]->weightMatrix += learn_rate * Wgrad[i];
				modules[i]->bias += learn_rate * bgrad[i];
			}
		}
		cout << "Ended epoch " << epoch + 1 << "/" << max_epoch << " of fine-tunning." << endl;

	}

	delete []features;
	delete []delta;
	//delete []features;
}

void UnsupervisedModel::predict(arma::mat& testdata, arma::imat& testlabels,vector<NewParam> params){
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
		cout << "Predict error:" << endl;
		cout << 100*(predict_acc(pred_labels,testlabels)) << "%" << endl;
	}

}

double UnsupervisedModel::predict_acc(const arma::imat predict_labels, const arma::imat labels){
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
