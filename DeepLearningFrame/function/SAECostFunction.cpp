#include "SAECostFunction.h"

void dlpft::function::SAECostFunction::initialParam(){
	coefficient.set_size(hiddenSize * visiableSize * 2 + hiddenSize + visiableSize);
	
	double r = sqrt(6) / sqrt(hiddenSize + visiableSize + 1);
	int h_v_size = hiddenSize * visiableSize;
	
	arma::mat W1 = arma::randu<arma::mat> (hiddenSize,visiableSize)*2*r-r;
	arma::mat W2 = arma::randu<arma::mat> (visiableSize,hiddenSize)*2*r-r;
	W1.reshape(h_v_size,1);
	W2.reshape(h_v_size,1);
	/*arma::mat W1 = arma::ones(hiddenSize,visiableSize)*2*r-r;
	arma::mat W2 = arma::ones(visiableSize,hiddenSize)*r-r;
	W1.reshape(h_v_size,1);
	W2.reshape(h_v_size,1);*/
	coefficient.rows(0,h_v_size-1) = W1;
	coefficient.rows(h_v_size,2*h_v_size-1) = W2;
	coefficient.rows(2*h_v_size,2 * h_v_size+hiddenSize-1) = arma::zeros(hiddenSize,1);
	coefficient.rows(2*h_v_size+hiddenSize,coefficient.size()-1) = arma::zeros(visiableSize,1);
	
}
double& dlpft::function::SAECostFunction::value_gradient(arma::mat& grad){
	/*clock_t start,end;
	double dur;
	start = clock();*/
	double cost = 0, Jcost = 0, Jweight = 0, Jsparse = 0;
	int n = data.n_rows;
	int m = data.n_cols;
	int h_v_size = hiddenSize * visiableSize;
	arma::mat W1(coefficient.rows(0,h_v_size-1));
	arma::mat W2(coefficient.rows(h_v_size,2*h_v_size-1));
	arma::vec b1(coefficient.rows(2*h_v_size,2*h_v_size+hiddenSize-1));
	arma::vec b2(coefficient.rows(2*h_v_size+hiddenSize,coefficient.size()-1));
	
	W1.reshape(hiddenSize,visiableSize);
	W2.reshape(visiableSize,hiddenSize);
/*
	end = clock();
	dur = (double)(end-start)/CLOCKS_PER_SEC;
	cout << "part 1:" << dur << "s" << endl;
	start = clock();*/

	fstream stream;

	arma::mat z2 = W1 * data + repmat(b1,1,m);
	arma::mat a2 = sigmoid(z2);
	arma::mat z3 = W2 * a2 + repmat(b2,1,m);
	arma::mat a3 = sigmoid(z3);
	/*
	
	end = clock();
	dur = (double)(end-start)/CLOCKS_PER_SEC;
	cout << "part 2:" << dur << "s" << endl;
	start = clock();		;*/
	arma::mat a3_x = pow((a3-data),2);

	Jcost = (0.5/m)*sum(sum(a3_x));
	
	Jweight = 0.5 * (arma::sum(arma::sum(arma::pow(W1,2))) + arma::sum(arma::sum(arma::pow(W2,2))));
	
	arma::vec rho = sum(a2,1)/m;
	
	Jsparse = sum(sparsityParam * log(sparsityParam/rho)+(1-sparsityParam) * log((1-sparsityParam)/(1-rho)));
	
	cost = Jcost+lambda*Jweight+beta*Jsparse;
	
/*
	end = clock();
	dur = (double)(end-start)/CLOCKS_PER_SEC;
	cout<< "part 3:"  << dur << "s" << endl;
	start = clock();*/

	//compute grad
	arma::mat W1grad(zeros(W1.n_rows,W1.n_cols));
	arma::mat W2grad(zeros(W2.n_rows,W2.n_cols));
	arma::vec b1grad(zeros(b1.size()));
	arma::vec b2grad(zeros(b2.size()));
	arma::mat d3 = -(data - a3) % sigmoidInv(z3);
	arma::mat sterm = beta * (-sparsityParam/rho + (1-sparsityParam)/(1-rho));
	arma::mat d2 = (W2.t()*d3 + repmat(sterm,1,m)) % sigmoidInv(z2);
	/*
	end = clock();
	dur = (double)(end-start)/CLOCKS_PER_SEC;
	cout<< "part 4:"  << dur << "s" << endl;
	start = clock();*/

	

	W1grad = W1grad + d2 * data.t();
	W1grad = W1grad/m + lambda * W1;

	

	W2grad = W2grad + d3*a2.t();
	W2grad = W2grad/m + lambda*W2;

	b1grad = b1grad + sum(d2,1);
	b1grad =  b1grad / m;
/*
	fstream fs;
	fs.open("B1.txt",fstream::out);
	b1grad.quiet_save(fs,arma::raw_ascii);
	fs.close();
*/
	b2grad = b2grad + sum(d3,1);

	b2grad = b2grad / m;

	

	


	W1grad.reshape(h_v_size,1);
	W2grad.reshape(h_v_size,1);

	grad = join_cols(W1grad,W2grad);
	grad = join_cols(grad,b1grad);
	grad = join_cols(grad,b2grad);
	
	/*
	end = clock();
	dur = (double)(end-start)/CLOCKS_PER_SEC;
	cout<< "part 5:"  << dur << "s" << endl;
	start = clock();*/
	return cost;
}
void dlpft::function::SAECostFunction::gradient(arma::mat& grad){

}
void dlpft::function::SAECostFunction::hessian(arma::mat& grad, arma::mat& hess){

}