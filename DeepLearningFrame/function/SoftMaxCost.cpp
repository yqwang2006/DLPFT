#include "SoftMaxCost.h"
#include "../util/onehot.h"

double dlpft::function::SoftMaxCost::value_gradient(arma::mat& grad){
	
	double cost = 0;
	double numCases = data.n_cols;
	
	arma::mat W = coefficient.rows(0,classesNum*visiableSize-1);
	W.reshape(classesNum,visiableSize);
	
	arma::mat bias = coefficient.rows(classesNum*visiableSize,coefficient.size()-1);
	arma::mat groundTruth = onehot(classesNum,numCases,labels);

	arma::mat Wgrad = zeros(classesNum,visiableSize);
	arma::mat bgrad = zeros(classesNum,1);

	//bxsfun(minus)
	arma::mat M = W * data + repmat(bias,1,data.n_cols);;
	M = active_function(SOFTMAXFUNC,M);
	
	Wgrad = ((double)-1/numCases) * ((groundTruth - M)) * data.t() + weightDecay * W;
	Wgrad.reshape(Wgrad.size(),1);

	bgrad = ((double)-1/numCases) * sum((groundTruth - M),1);
	
	
	groundTruth.reshape(groundTruth.size(),1);
	M.reshape(M.size(),1);
	W.reshape(W.size(),1);
	
	double gm_cost = dot(groundTruth,log(M));

	cost = ((double)-1/numCases)*gm_cost;

	cost += weightDecay/2 * sum(sum(pow(W,2)));
	
	grad.set_size(Wgrad.size()+bgrad.size(),1);
	grad.rows(0,Wgrad.size()-1) = reshape(Wgrad,Wgrad.size(),1);
	grad.rows(Wgrad.size(),grad.size()-1) = reshape(bgrad,bgrad.size(),1);

	return cost;

}

void dlpft::function::SoftMaxCost::gradient(arma::mat& grad){

}
void dlpft::function::SoftMaxCost::hessian(arma::mat& grad, arma::mat& hess){

}