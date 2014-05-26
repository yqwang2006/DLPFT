#ifndef SOFTMAXCOST_H
#define SOFTMAXCOST_H
#include "CostFunction.h"
using namespace arma;
namespace dlpft{
	namespace function{
		class SoftMaxCost
			:public CostFunction
		{
		private:
			arma::imat labels;
			int visiableSize;
			int classesNum;
			double lambda;
		public:
			SoftMaxCost(void):CostFunction(){
				initialParam();
				function_name = "softmax function";
				cout << function_name << endl;
			}


			SoftMaxCost(int v, int c,const arma::mat d,const arma::imat  l, const double lambda=3e-3,
				const string func_name = "softmax function")
				:visiableSize(v),classesNum(c),lambda(lambda){
					labels = l;
					data = d;
					initialParam();
					function_name = func_name;
			}


			~SoftMaxCost(void){
			}

			/*get and set*/
			arma::mat get_coefficient(){return coefficient;}
			arma::imat get_labels(){return labels;}

			int get_visiableSize() const {return visiableSize;}
			void set_visiableSize(int& v){visiableSize = v;}
			int get_hiddenSize() const{return classesNum;}
			void set_hiddenSize(int& h) { classesNum = h;}
			double get_lambda() const {return lambda;}
			void set_lambda(double& l) {lambda = l;}


			void initialParam();
			double value_gradient(arma::mat& grad);
			void gradient(arma::mat& grad);
			void hessian(arma::mat& grad, arma::mat& hess);
		};
	};
};
#endif