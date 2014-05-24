#pragma once
#include "armadillo"
#include <vector>
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
		public:
			CostFunction(void):function_name(""){}
			~CostFunction(void){
				cout << "~CostFunction" << endl;
			}
			virtual double value_gradient(arma::mat&){
				double a = 0;
				return a;
			}
			virtual void gradient(arma::mat&){}
			virtual void hessian(arma::mat&, arma::mat&){}
			arma::mat get_coefficient(){return coefficient;}
			void set_coefficient(const arma::mat the){ coefficient = the;}
			arma::mat get_data() const{return data;}
			void set_data(const arma::mat d){ data = d;}
			string get_func_name(){return function_name;}
		};
		
	};
};
