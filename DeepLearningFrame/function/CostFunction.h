#pragma once
#include "armadillo"
#include <vector>
#include "../util/ActiveFunction.h"
#include <string>
using namespace std;
namespace dlpft{
	namespace function{
		
		class CostFunction
		{
		public:
			string function_name;
			arma::mat coefficient;
			arma::mat data;
			arma::mat labels;
			double inputZeroMaskedFraction;
			ActivationFunction activeFuncChoice;
		public:
			CostFunction(void):function_name(""){
				activeFuncChoice = SIGMOIDFUNC;
			}
			~CostFunction(void){
				cout << "~CostFunction" << endl;
			}
			virtual double value_gradient(arma::mat&){
				double a = 0;
				return a;
			}
			virtual void gradient(arma::mat&)=0;
			virtual void hessian(arma::mat&, arma::mat&) = 0;
		};
		
	};
};
