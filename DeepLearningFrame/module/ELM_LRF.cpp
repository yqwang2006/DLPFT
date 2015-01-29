#include "ELM_LRF.h"
#include "../util/onehot.h"
using namespace dlpft::param;
using namespace dlpft::module;
void ELM_LRF::train(const arma::mat data, const arma::mat labels, NewParam param){
	
	int hid_size = atoi(param.params[params_name[HIDNUM]].c_str());
	double C = atof(param.params[params_name[ELM_C]].c_str());
	string elm_type = param.params[params_name[ELMTYPE]].c_str();
	int class_num = atoi(param.params[params_name[ELMCLASSNUM]].c_str());
	
	int numberofsamples = data.n_cols;

	mat T = zeros(class_num, numberofsamples);
	if (elm_type != "REGESSION"){

		T = onehot_elm(class_num, numberofsamples ,labels);

		mat H = data;

		outputWeight = inv(eye(H.n_rows, H.n_rows)/C+H*H.t()) * H * T.t();

		mat output = (H.t() * outputWeight).t();

		mat max_vals; max_vals = max(output);

		mat pred_labels = zeros<arma::mat>(numberofsamples,1);

		for(int i = 0;i < output.n_cols;i++){
			pred_labels[i] = 0;
			for(int j = 0;j < output.n_rows; j++){
				if(max_vals(i) == output(j,i)){
					pred_labels[i] = j+1;
					continue;
				}
			}

		}

		int sum = 0;
		for(int i = 0;i < pred_labels.size();i++){
			if(pred_labels(i) == labels(i)){
				sum ++;
			}
		}

		cout << "TrainSet Accuracy: " << (double)sum / numberofsamples << endl;

		LogOut << "TrainSet Accuracy: " << (double)sum / numberofsamples << endl;
	
	}

}

void ELM_LRF::initial_weights_bias(){
		srand(unsigned(time(NULL)));
		weightMatrix = randu(outputSize,inputSize)*2-1;
		bias = arma::randu(outputSize,1);
	
}
arma::mat ELM_LRF::forwardpropagate(const mat data,  NewParam param){
	arma::mat activation = weightMatrix * data + repmat(bias,1,data.n_cols);
	activation = active_function(activeFuncChoice,activation);
	return activation;
}

