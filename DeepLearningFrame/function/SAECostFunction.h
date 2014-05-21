#ifndef SAECOST_H
#define SAECOST_H
#include "CostFunction.h"
#include "../util/sigmoid.h" 
using namespace arma;
namespace dlpft{
	namespace function{
		class SAECostFunction
			:public CostFunction
		{
		private:
			int visiableSize;
			int hiddenSize;
			double lambda;
			double sparsityParam;
			double beta;
		public:
			SAECostFunction(void):CostFunction(){
				initialParam();
				function_name = "sparse autoencoder function";
				cout << function_name << endl;
			}


			SAECostFunction(int v, int h, const double lambda=3e-3,
				const double sparsityParam=0.1,const double beta = 3e-3,const string func_name = "sparse autoencoder function")
				:visiableSize(v),hiddenSize(h),lambda(lambda),sparsityParam(sparsityParam),beta(beta){
					coefficient.set_size(v*h*2+v+h,1);
					initialParam();
					function_name = func_name;
			}
			~SAECostFunction(void){
			}

			/*get and set*/
			arma::mat get_coefficient(){return coefficient;}


			int get_visiableSize() const {return visiableSize;}
			void set_visiableSize(int v){visiableSize = v;}
			int get_hiddenSize() const{return hiddenSize;}
			void set_hiddenSize(int h) { hiddenSize = h;}
			double get_lambda() const {return lambda;}
			void set_lambda(double l) {lambda = l;}
			double get_beta() const{return beta;}
			void set_beta(double b){ beta = b;}
			double get_sparsityParam() const{return sparsityParam;}
			void set_sparsityParam(double s){ sparsityParam = s;}


			void initialParam();
			double& value_gradient(arma::mat& grad);
			void gradient(arma::mat& grad);
			void hessian(arma::mat& grad, arma::mat& hess);
		};
	};
};
#endif