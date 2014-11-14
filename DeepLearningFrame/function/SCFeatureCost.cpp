#include "SCFeatureCost.h"
using namespace dlpft::function;
double SCFeatureCost::value_gradient(arma::mat& grad){
	int samples_num = data.n_cols;
	double Jcost = 0;
	weightMatrix.reshape(hidden_size,visible_size);
	coefficient.reshape(hidden_size,samples_num);
	
	arma::mat delta = weightMatrix.t() * coefficient - data;

	double Jcost1 = accu(pow(delta,2))/samples_num;
	arma::mat sparsityMat = sqrt(group_matrix*(pow(coefficient,2))+epsilon);
	double Jsparse = lambda * accu(sparsityMat);
	//double Jweight = gamma * sum(sum(pow(weightMatrix,2)));

	Jcost = Jcost1 + Jsparse ;//+ Jweight;

	
	arma::mat grad1 = ((double)1/samples_num) * (-2*weightMatrix * data + 2 * weightMatrix * weightMatrix.t() * coefficient);
	arma::mat grad2 = arma::zeros(hidden_size,samples_num);
	if(is_topo){
		grad2 = lambda * group_matrix.t()*pow(group_matrix*coefficient.t()*coefficient+epsilon,-0.5)%coefficient;
	}else{
		grad2 = lambda * coefficient / (sparsityMat);
	}


	grad = grad1 + grad2;
	

	grad.reshape(grad.size(),1);
	coefficient.reshape(coefficient.size(),1);
	return Jcost;
}
void SCFeatureCost::gradient(arma::mat& grad){

}
void SCFeatureCost::hessian(arma::mat& grad, arma::mat& hess){

}