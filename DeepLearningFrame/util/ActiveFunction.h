#ifndef ACTIVEFUNCTION_H
#define ACTIVEFUNCTION_H

#include "armadillo"
#include <xmmintrin.h>

	enum ActivationFunction {LINEARFUNC,TANHFUNC,RECTIFIERFUNC,SIGMOIDFUNC,SOFTMAXFUNC};
	static ActivationFunction get_activation_function(std::string name){
		if(name == "LINEAR"){
			return LINEARFUNC;
		}else if(name == "TANH"){
			return TANHFUNC;
		}else if(name == "RECTIFIER"){
			return RECTIFIERFUNC;
		}else if(name == "SOFTMAX"){
			return SOFTMAXFUNC;
		}else{
			return SIGMOIDFUNC;
		}
	}

	inline static void sigmoid(const arma::mat x, arma::mat& y){
		y = 1/(1+arma::exp(-x));
	}
	static void tanh(const arma::mat x, arma::mat& y){
		y = arma::tanh(x);
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
		g = t+arma::pow(z,2);
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
		case SIGMOIDFUNC:
			sigmoid(a,z);
			break;
		case TANHFUNC:
			tanh(a,z);
			break;
		case RECTIFIERFUNC:
			rectifier(a,z);
			break;
		case LINEARFUNC:
			linear(a, z);
			break;
		case SOFTMAXFUNC:
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
		case SIGMOIDFUNC:
			sigmoid_dev(z, g);
			break;
		case TANHFUNC:
			tanh_dev(z, g);
			break;
		case RECTIFIERFUNC:
			rectifier_dev(z, g);
			break;
		case LINEARFUNC:
			linear_dev(g);
			break;
		case SOFTMAXFUNC:
			softmax_dev(z,g);
			break;
		default:
			sigmoid_dev(z,g);
			break;
		}
		return g;
	}



#endif