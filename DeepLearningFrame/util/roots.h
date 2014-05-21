#ifndef ROOTS_H
#define ROOTS_H
#include "armadillo"
static arma::cx_vec roots(arma::vec& v){
	
	int order = v.size() - 1;
	arma::mat A1 = -v.rows(1,order)/v(0);
	A1 = A1.t();
	arma::mat A2 = arma::join_horiz(arma::eye(order-1,order-1),arma::zeros(order-1,1));

	arma::mat A = arma::join_cols(A1,A2);
	
	arma::cx_vec r = eig_gen(A);
	
	return r;
}
static std::complex<double> polyval(arma::vec& p, std::complex<double> x){
	std::complex<double> result = 0;
	int order = p.size()-1;
	for(int i = 0;i < order + 1;i++){
		result += p(i)*pow(x,order-i);
	}
	return result;
} 

#endif