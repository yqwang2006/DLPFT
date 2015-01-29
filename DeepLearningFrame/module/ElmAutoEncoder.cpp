#include "ElmAutoEncoder.h"
using namespace dlpft::param;
using namespace dlpft::module;
void ElmAutoEncoder::pretrain(const arma::mat data, NewParam param){

	int hid_size = atoi(param.params[params_name[HIDNUM]].c_str());

	double C = atof(param.params[params_name[ELM_C]].c_str());

	int numberofsamples = data.n_cols;

	mat H = forwardpropagate(data, param);

	outputWeight = inv(eye(H.n_rows, H.n_rows)/C+H*H.t()) * H * data.t();

	weightMatrix = outputWeight;

	mat output = (H.t() * outputWeight).t();

	cout << "Reconstruction error: " << accu((output-data)%(output-data))/data.n_elem << endl;

	LogOut << "Reconstruction error: " << accu(pow(output-data,2))/data.n_elem << endl;



}

void ElmAutoEncoder::initial_weights_bias(){
	srand(unsigned(time(NULL)));
	weightMatrix = randu(outputSize,inputSize)*2-1;
	bias = arma::randu(outputSize,1);

}
arma::mat ElmAutoEncoder::forwardpropagate(const mat data,  NewParam param){
	arma::mat activation = weightMatrix * data + repmat(bias,1,data.n_cols);
	activation = active_function(activeFuncChoice,activation);
	return activation;
}

