#include "UnsupervisedModel.h"
using namespace dlpft::factory;
using namespace dlpft::model;
ResultModel* UnsupervisedModel::pretrain(const arma::mat data,const arma::imat labels, vector<NewParam> params){
	int number_layer = params.size();

	ResultModel *resultmodel_ptr = new ResultModel[number_layer];
	arma::mat features = data;
	Module* single_module;

	for(int i = 0;i < number_layer;i++){
		single_module = create_module(params[i]);
		resultmodel_ptr[i] = single_module->pretrain(features,labels,params[i]);
		features = single_module->forwardpropagate(resultmodel_ptr[i],features,labels,params[i]);
	}

	if(finetune_switch){
		assert(params[number_layer-1].params[params_name[ALGORITHM]] == "SoftMax");
		finetune_BP(resultmodel_ptr,data,labels,params);
	}

	delete single_module;
	single_module = NULL;
	return resultmodel_ptr;
}
void UnsupervisedModel::finetune_BP(ResultModel* result_model_ptr,const arma::mat data, const arma::imat labels, vector<NewParam> params){
	int number_layer = params.size();
	int samples_num = data.n_cols;
	arma::mat* features = new arma::mat[number_layer];
	arma::mat* delta = new arma::mat[number_layer+1];
	Module** single_module = new Module*[number_layer];
	ResultModel* prev_result_model = new ResultModel[number_layer];
	for(int i = 0;i < number_layer;i++){
		prev_result_model[i] = result_model_ptr[i];
	}

	int max_epoch = 5;

	for(int i = 0;i < number_layer ;i++){
		single_module[i] = create_module(params[i]);
	}

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
					features[i] = single_module[i]->forwardpropagate(result_model_ptr[i],minibatch,minibatch_labels,params[i]);
				}else{
					features[i] = single_module[i]->forwardpropagate(result_model_ptr[i],features[i-1],minibatch_labels,params[i]);
				}
			}

			arma::mat desired_out = zeros(features[number_layer-1].n_rows,features[number_layer-1].n_cols);
			for(int i = 0;i < features[number_layer-1].n_cols; i++){
				if(minibatch_labels(i) == features[number_layer-1].n_rows)
					desired_out(0,i) = 1;
				else
					desired_out(minibatch_labels(i),i) = 1;
			}
			// compute the output delta
			delta[number_layer] = (features[number_layer-1] % (1-features[number_layer-1]))%(desired_out - features[number_layer-1]);

			for(int i = number_layer-1;i >0 ;i--){	
				delta[i] = single_module[i]->backpropagate(result_model_ptr[i],delta[i+1],features[i-1],minibatch_labels,params[i]);
			}

			for(int i = 0;i < number_layer; i++){
				double learn_rate = atof(params[i].params["Learning_rate"].c_str());

				if(i == 0){
					prev_result_model[i].weightMatrix = learn_rate * delta[i+1]*minibatch.t()/batch_size;
					prev_result_model[i].bias = (learn_rate / batch_size) * sum(delta[i+1],1);
				}else{
					prev_result_model[i].weightMatrix = learn_rate * delta[i+1]*features[i-1].t()/batch_size;
					prev_result_model[i].bias = (learn_rate / batch_size) * sum(delta[i+1],1);
				}
				result_model_ptr[i].weightMatrix += prev_result_model[i].weightMatrix;
				if(i != number_layer - 1){
					result_model_ptr[i].bias += prev_result_model[i].bias;
				}
			}
		}
		cout << "Ended epoch " << epoch + 1 << "/" << max_epoch << " of fine-tunning." << endl;

	}

	for(int i = 0;i < number_layer;i++){
		delete single_module[i];
	}
	delete []single_module;
	delete []features;
	delete []delta;
	delete []prev_result_model;
	//delete []features;
	single_module = NULL;
}
