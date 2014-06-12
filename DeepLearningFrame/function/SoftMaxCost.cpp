#include "SoftMaxCost.h"


double dlpft::function::SoftMaxCost::value_gradient(arma::mat& grad){
	
	
	arma::mat W = coefficient.rows(0,classesNum*visiableSize-1);
	W.reshape(classesNum,visiableSize);
	arma::mat bias = coefficient.rows(classesNum*visiableSize,coefficient.size()-1);
	double numCases = data.n_cols;
	arma::mat groundTruth = zeros(classesNum,numCases);
	for(int i = 0;i < numCases; i++){
		if(labels(i) == classesNum)
			groundTruth(0,i) = 1;
		else
			groundTruth(labels(i),i) = 1;
	} 
	double cost = 0;
	arma::mat Wgrad = zeros(classesNum,visiableSize);
	arma::mat bgrad = zeros(classesNum,1);

	//bxsfun(minus)
	arma::mat M = W * data + repmat(bias,1,data.n_cols);  //10 * 5000
	//arma::mat M = W * data;  //10 * 5000
	arma::mat max_M = max(M,0);//1*5000
	arma::mat sum_M = sum(M,0);
	for(int i = 0;i < M.n_rows;i++){
		M.row(i) = exp(M.row(i)-max_M);
	}
	max_M = sum(M,0);
	for(int i = 0;i < M.n_rows;i++){
		M.row(i) = M.row(i)/max_M;
	}

//active_function_dev(activeFuncChoice,M)%
	Wgrad = ((double)-1/numCases) * ((groundTruth - M)) * data.t() + lambda * W;
	Wgrad.reshape(Wgrad.size(),1);

	bgrad = ((double)-1/numCases) * sum((groundTruth - M),1);

	groundTruth.reshape(1,groundTruth.size());
	M.reshape(M.size(),1);
	W.reshape(W.size(),1);
	arma::mat gm = -1*groundTruth * log(M)/numCases;
	cost = gm(0);
	cost += lambda/2*sum(sum((pow(W,2))));
	
	grad.set_size(Wgrad.size()+bgrad.size(),1);
	//grad.set_size(Wgrad.size(),1);
	grad.rows(0,Wgrad.size()-1) = reshape(Wgrad,Wgrad.size(),1);
	grad.rows(Wgrad.size(),grad.size()-1) = reshape(bgrad,bgrad.size(),1);

	return cost;

}

void dlpft::function::SoftMaxCost::gradient(arma::mat& grad){

}
void dlpft::function::SoftMaxCost::hessian(arma::mat& grad, arma::mat& hess){

}