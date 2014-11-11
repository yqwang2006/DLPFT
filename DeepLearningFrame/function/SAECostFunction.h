#ifndef SAECOST_H
#define SAECOST_H
#include "CostFunction.h"
using namespace arma;
namespace dlpft{
	namespace function{
		class SAECostFunction
			:public CostFunction
		{
		private:
			int visiableSize;
			int hiddenSize;
			double weight_decay;
			double kl_rho;
			double sparsity;
			
		public:
			SAECostFunction(void):CostFunction(){
				function_name = "sparse autoencoder function";
				cout << function_name << endl;
			}


			SAECostFunction(int v, int h,double inputZeroFraction,const double sparsity = 3e-3, const double weight_decay=3e-3,
				const double kl_rho_dist=0.05,const string func_name = "sparse autoencoder function")
				:visiableSize(v),hiddenSize(h),weight_decay(weight_decay),kl_rho(kl_rho_dist),sparsity(sparsity){
					inputZeroMaskedFraction = inputZeroFraction;
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
			double get_weight_decay() const {return weight_decay;}
			void set_weight_decay(double l) {weight_decay = l;}
			double get_sparsity() const{return sparsity;}
			void set_sparsity(double b){ sparsity = b;}
			double get_kl_rho() const{return kl_rho;}
			void set_kl_rho(double s){ kl_rho = s;}


			double value_gradient(arma::mat& grad);
			void gradient(arma::mat& grad);
			void hessian(arma::mat& grad, arma::mat& hess);
		};
	};
};
#endif