#pragma once
#include "CostFunction.h"
namespace dlpft{
	namespace function{
		class TestFunction
			:public CostFunction
		{
		private:
			arma::mat data;
		public:
			TestFunction(void){
				function_name = "test function";
				cout << function_name << endl;
			}

			TestFunction(string func_name){
					function_name = func_name;
			}

			~TestFunction(void){}
			double value_gradient(arma::mat& grad){
				double x1 = data[0];
				double x2 = data[1];
				double obj = 100 * std::pow(x2-std::pow(x1,2),2) + std::pow(1-x1,2);
				return obj;
			}
			void gradient(arma::mat& grad){
				double x1 = data[0];
				double x2 = data[1];
				grad.set_size(2,1);
				grad[0] = -2 * (1-x1) + 400 * (std::pow(x1,3)-x2*x1);
				grad[1] = 200 * (x2 - std::pow(x1,2));
				
			}
			void hessian(arma::mat& grad, arma::mat& hes){
				
			}
			
		};
	};
};
