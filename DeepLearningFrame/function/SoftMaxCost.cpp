#include "SoftMaxCost.h"

void dlpft::function::SoftMaxCost::initialParam(){
	//coefficient->set_size(visiableSize*classesNum,1);
	coefficient = 0.005*arma::randu<arma::mat> (classesNum,visiableSize);
	//coefficient = 0.005* arma::ones(classesNum*visiableSize,1);
}

double dlpft::function::SoftMaxCost::value_gradient(arma::mat& grad){
	
	
	coefficient.reshape(classesNum,visiableSize);
	double numCases = data.n_cols;
	arma::mat groundTruth = zeros(classesNum,numCases);
	for(int i = 0;i < numCases; i++){
		if(labels(i) == numCases)
			groundTruth(0,i) = 1;
		else
			groundTruth(labels(i),i) = 1;
	}
	double cost = 0;
	grad = zeros(classesNum,visiableSize);


	//bxsfun(minus)
	arma::mat M = coefficient * data;  //10 * 5000
	arma::mat max_M = max(M,0);//1*5000
	arma::mat sum_M = sum(M,0);
	for(int i = 0;i < M.n_rows;i++){
		M.row(i) = exp(M.row(i)-max_M);
	}
	max_M = sum(M,0);
	for(int i = 0;i < M.n_rows;i++){
		M.row(i) = M.row(i)/max_M;
	}


	grad = -1/numCases * (groundTruth - M) * data.t() + lambda * coefficient;
	grad.reshape(grad.size(),1);


	groundTruth.reshape(1,groundTruth.size());
	M.reshape(M.size(),1);
	coefficient.reshape(coefficient.size(),1);
	arma::mat gm = -1*groundTruth * log(M)/numCases;
	cost = gm(0);
	cost += lambda/2*sum(sum((pow(coefficient,2))));
	

	return cost;

}

void dlpft::function::SoftMaxCost::gradient(arma::mat& grad){

}
void dlpft::function::SoftMaxCost::hessian(arma::mat& grad, arma::mat& hess){

}