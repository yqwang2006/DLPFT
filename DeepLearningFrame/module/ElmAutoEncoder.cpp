#include "ElmAutoEncoder.h"
using namespace dlpft::param;
using namespace dlpft::module;
void ElmAutoEncoder::pretrain(const arma::mat data, NewParam param){

	int hid_dim = atoi(param.params[params_name[HIDNUM]].c_str());

	double C = atof(param.params[params_name[ELM_C]].c_str());

	double rho = atof(param.params[params_name[KLRHO]].c_str());
	
	int numberofsamples = data.n_cols;

	int input_dim = data.n_rows;


	mat H = forwardpropagate(data, param);

	if ( input_dim == hid_dim ){
		
		outputWeight = procrustNew(data.t(), H.t());

	}else{

		if( C==0 ){
			outputWeight = pinv(H.t()) * data.t();
		}else{

			vec rho_hat = mean(H,1);
			double KLsum = sum( rho * log(rho / rho_hat)+(1-rho) * log((1-rho) / (1-rho_hat)));
			cout << KLsum << endl;
			mat Hsquare = H * H.t();
			mat HsquareL = diagmat(max(Hsquare,1));
			outputWeight = (inv((eye(H.n_rows,H.n_rows)*KLsum + HsquareL)/C+Hsquare)) * H * data.t();
		}
	}

	weightMatrix = outputWeight;

	mat output = (H.t() * outputWeight).t();

	cout << data.n_elem << endl;

	cout << accu((output-data)%(output-data)) << endl;

	cout << "RMSE: " << sqrt(accu((output-data)%(output-data))/data.n_elem) << endl;

	LogOut << "RMSE: " << sqrt(accu(pow(output-data,2))/data.n_elem) << endl;



}
mat ElmAutoEncoder::procrustNew(mat A, mat B){
	mat C = B.t() * A;
	mat U1,U2, V1,V2;
	vec s1, s2;
	svd(U1,s1,V1,C.t()*C);
	svd(U2,s2,V2,C*C.t());

	mat Q = U2 * V1.t();

	return Q;
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

