#pragma once
#include "CostFunction.h"
namespace dlpft{
	namespace function{
		class ScWeightFunction
			:public CostFunction
		{
		private:
			arma::mat theta;
			int visiableSize;
			int hiddenSize;
			double lambda;
			double kl_rho;
			double beta;
		public:
			ScWeightFunction(void){
				function_name = "sparse coding weight function";
				cout << function_name << endl;
			}

			ScWeightFunction(arma::mat theta,int v, int h, const double lambda=0.001,
				const double kl_rho=0.001,const double beta = 0.001,const string func_name = "sparse coding weight function")
				:theta(theta),visiableSize(v),hiddenSize(h),lambda(lambda),kl_rho(kl_rho),beta(beta){
					function_name = func_name;
					
			}

			~ScWeightFunction(void){}
			double value_gradient(arma::mat& grad){
				
				return 0;
			}
			void gradient(arma::mat& grad){
				
			}
			void hessian(arma::mat& grad, arma::mat& hess){
			
			}
		};
	};
};
