#include "SAECostFunction.h"


double dlpft::function::SAECostFunction::value_gradient(arma::mat& grad){
	/*clock_t start,end;
	double dur;
	start = clock();*/
	arma::mat a2,z2,a3,z3;
	arma::mat noiseData = data;
	if(inputZeroMaskedFraction > 0){
		arma::mat rand_mat = arma::randu(data.n_rows,data.n_cols);
		arma::uvec indeies = find(data<=rand_mat);
		noiseData(indeies) = zeros(indeies.size());
		
	}
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

	if(inputZeroMaskedFraction > 0){
		z2 = W1 * noiseData + repmat(b1,1,m);
	}else{
		z2 = W1 * data + repmat(b1,1,m);
	}
	a2 = active_function(activeFuncChoice,z2);
	z3 = W2 * a2 + repmat(b2,1,m);
	a3 = active_function(activeFuncChoice,z3);


	arma::mat a3_x = pow((a3-data),2);

	Jcost = (0.5/m)*sum(sum(a3_x));
	
	Jweight = 0.5 * (arma::sum(arma::sum(arma::pow(W1,2))) + arma::sum(arma::sum(arma::pow(W2,2))));
	
	arma::vec rho = sum(a2,1)/m;
	
	Jsparse = sum(kl_rho * log(kl_rho/rho)+(1-kl_rho) * log((1-kl_rho)/(1-rho)));
	
	cost = Jcost+weight_decay*Jweight+sparsity*Jsparse;
	
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
	arma::mat d3 = -(data - a3) % active_function_dev(activeFuncChoice,a3);
	arma::mat sterm = sparsity * (-kl_rho/rho + (1-kl_rho)/(1-rho));
	arma::mat d2 = (W2.t()*d3 + repmat(sterm,1,m)) % active_function_dev(activeFuncChoice,a2);
	/*
	end = clock();
	dur = (double)(end-start)/CLOCKS_PER_SEC;
	cout<< "part 4:"  << dur << "s" << endl;
	start = clock();*/

	

	W1grad = W1grad + d2 * data.t();
	W1grad = W1grad/m + weight_decay * W1;

	

	W2grad = W2grad + d3*a2.t();
	W2grad = W2grad/m + weight_decay*W2;

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