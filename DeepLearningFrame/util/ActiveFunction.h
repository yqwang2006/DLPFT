#ifndef ACTIVEFUNCTION_H
#define ACTIVEFUNCTION_H

#include "armadillo"

namespace dlpft{

	enum ActivationFunction {SIGMOID, TANH, RECTIFIER, LINEAR};


	static void sigmoid(const arma::mat& x, arma::mat& y){
		y = 1/(1+arma::exp(-x));
	}
	static void tanh(const arma::mat& x, arma::mat& y){
		y = 1/(1+arma::tanh(-x));
	}
	static void rectifier(const arma::mat& x, arma::mat& y){
		for(int i = 0;i < x.size();i++){
			y(i) = std::max<double>(0.0,x(i));
		}
	}
	static void linear(const arma::mat&x, arma::mat& y){
		y = x;
	}
	static void sigmoid_inv(const arma::mat& z, arma::mat& g){
		g = z % (1-z);
	}
	static void tanh_inv(const arma::mat& z, arma::mat& g){
		arma::mat t = arma::ones(z.n_rows,z.n_cols);
		g = t-arma::pow(z,2);
	}
	static void rectifier_inv(const arma::mat& z, arma::mat& g){
		for(int i = 0;i < z.size();i++){
			g(i) =(double)(z(i) > 0.0)*1.0;
		}
	}
	static void linear_inv( arma::mat& g){
		g.fill(1.0);
	}

		static arma::mat active_function(ActivationFunction act, const arma::mat& a){
		arma::mat z = arma::zeros(a.n_rows,a.n_cols);
		switch(act)
		{
		case SIGMOID:
			sigmoid(a,z);
			break;
		case TANH:
			tanh(a,z);
			break;
		case RECTIFIER:
			rectifier(a,z);
			break;
		case LINEAR:
			linear(a, z);
			break;
		default:
			sigmoid(a,z);
			break;
		}
		return z;
	}
	static arma::mat active_function_inv(ActivationFunction act, const arma::mat& z){
		arma::mat g = arma::zeros(z.n_rows,z.n_cols);
		switch(act)
		{
		case SIGMOID:
			sigmoid_inv(z, g);
			break;
		case TANH:
			tanh_inv(z, g);
			break;
		case RECTIFIER:
			rectifier_inv(z, g);
			break;
		case LINEAR:
			linear_inv(g);
			break;
		default:
			sigmoid_inv(z,g);
			break;
		}
		return g;
	}

};


#endif