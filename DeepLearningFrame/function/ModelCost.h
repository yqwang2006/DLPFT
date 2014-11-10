#ifndef MODELCOST_H
#define MODELCOST_H
#include "CostFunction.h"
#include "../param/NewParam.h"
#include "../module/AllModule.h"
using namespace arma;
using namespace dlpft::param;
namespace dlpft{
	namespace function{
		class ModelCost
			:public CostFunction
		{
		public:
			vector<NewParam> params;
			Module** modules;
			int layer_num;
			double weight_decay;
		public:
			ModelCost(void):CostFunction(){
				function_name = "Model cost function";
				cout << function_name << endl;
			}


			ModelCost(Module** m,arma::mat d , arma::mat l,vector<NewParam> np,const double weight_dec = 3e-3,
				const string func_name = "Model cost function")
			{
					labels = l;
					data = d;
					modules = m;
					layer_num = np.size()-1;
					params = np;
					weight_decay = weight_dec;
					if(weight_decay == 0) weight_decay = 3e-3;
					initialParam();
					function_name = func_name;
					
	
			}


			~ModelCost(void){
			}

			/*get and set*/


			void initialParam();
			double value_gradient(arma::mat& grad);
			
			void modelff(const arma::mat inputdata,arma::mat *output,arma::mat* dropoutMask);
			double modelbp(const arma::mat* features,arma::mat* dropoutMask,arma::mat *outputdelta,const int num_samples);

			void gradient(arma::mat& grad);

			void hessian(arma::mat& grad, arma::mat& hess);
			void paramsToStack();
		};
	};
};
#endif