#include "CNNCost.h"
using namespace dlpft::module;
using namespace dlpft::param;
using namespace dlpft::function;
void CNNCost::initialParam(){
	layer_num = params.size();
	for(int i = 0;i < layer_num;i++){
		
	}
}
double CNNCost::value_gradient(arma::mat& grad){
	

	return 0;
}
void CNNCost::gradient(arma::mat& grad){

}
void CNNCost::hessian(arma::mat& grad, arma::mat& hess){

}