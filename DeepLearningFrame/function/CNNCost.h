#ifndef CNNCOST_H
#define CNNCOST_H
#include "CostFunction.h"
#include "../param/NewParam.h"
#include "../module/AllModule.h"
using namespace arma;
using namespace dlpft::param;
namespace dlpft{
	namespace function{
		class CNNCost
			:public CostFunction
		{
		public:
			arma::imat labels;
			vector<NewParam> params;
			Module** modules;
			int layer_num;
			
		public:
			CNNCost(void):CostFunction(){
				function_name = "cnn cost function";
				cout << function_name << endl;
			}


			CNNCost(Module** m,arma::mat d , arma::imat l,vector<NewParam> np,
				const string func_name = "cnn cost function")
			{
					labels = l;
					data = d;
					modules = m;
					layer_num = np.size();
					initialParam();
					function_name = func_name;
			}


			~CNNCost(void){
			}

			/*get and set*/


			void initialParam();
			double value_gradient(arma::mat& grad);
			void gradient(arma::mat& grad);
			void hessian(arma::mat& grad, arma::mat& hess);
			void cnnParamsToStack();
		};
	};
};
#endif