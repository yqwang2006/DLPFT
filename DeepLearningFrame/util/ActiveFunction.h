#ifndef ACTIVEFUNCTION_H
#define ACTIVEFUNCTION_H

#include "armadillo"

namespace dlpft{

	enum ActivationFunction {SIGMOID, TANH, RECTIFIER, LINEAR,SOFTMAX};


	static void sigmoid(const arma::mat x, arma::mat& y){
		y = 1/(1+arma::exp(-x));
	}
	static void tanh(const arma::mat x, arma::mat& y){
		y = 1/(1+arma::tanh(-x));
	}
	static void rectifier(const arma::mat x, arma::mat& y){
		for(int i = 0;i < x.size();i++){
			y(i) = std::max<double>(0.0,x(i));
		}
	}
	static void linear(const arma::mat x, arma::mat& y){
		y = x;
	}
	static void sigmoid_dev(const arma::mat z, arma::mat& g){
		g = z % (1-z);
	}
	static void tanh_dev(const arma::mat z, arma::mat& g){
		arma::mat t = arma::ones(z.n_rows,z.n_cols);
		g = t-arma::pow(z,2);
	}
	static void rectifier_dev(const arma::mat z, arma::mat& g){
		for(int i = 0;i < z.size();i++){
			g(i) =(double)(z(i) > 0.0)*1.0;
		}
	}
	static void linear_dev( arma::mat& g){
		g.fill(1.0);
	}
	static void softmax(const arma::mat x, arma::mat& features){
		arma::mat max_M = max(x,0);//1*5000
		for(int i = 0;i < x.n_rows;i++){
			features.row(i) = exp(x.row(i)-max_M);
		}
		max_M = sum(features,0);
		for(int i = 0;i < x.n_rows;i++){
			features.row(i) = features.row(i)/max_M;
		}
	}
	static void softmax_dev(const arma::mat z, arma::mat& g){
		g = z % (1-z);
	}

	static arma::mat active_function(ActivationFunction act, const arma::mat a){
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
		case SOFTMAX:
			softmax(a,z);
			break;
		default:
			sigmoid(a,z);
			break;
		}
		return z;
	}
	static arma::mat active_function_dev(ActivationFunction act, const arma::mat z){
		arma::mat g = arma::zeros(z.n_rows,z.n_cols);
		switch(act)
		{
		case SIGMOID:
			sigmoid_dev(z, g);
			break;
		case TANH:
			tanh_dev(z, g);
			break;
		case RECTIFIER:
			rectifier_dev(z, g);
			break;
		case LINEAR:
			linear_dev(g);
			break;
		case SOFTMAX:
			softmax_dev(z,g);
			break;
		default:
			sigmoid_dev(z,g);
			break;
		}
		return g;
	}

};


#endif