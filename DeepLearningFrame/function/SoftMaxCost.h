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
			int visiableSize;
			int classesNum;
			double weightDecay;
		public:
			SoftMaxCost(void):CostFunction(){
				function_name = "softmax function";
				cout << function_name << endl;
			}


			SoftMaxCost(int v, int c,const arma::mat d,const arma::mat  l, const double lambda=3e-3,
				const string func_name = "softmax function")
				:visiableSize(v),classesNum(c),weightDecay(lambda){
					labels = l;
					data = d;
					function_name = func_name;
			}


			~SoftMaxCost(void){
			}

			/*get and set*/
			arma::mat get_coefficient(){return coefficient;}
			arma::mat get_labels(){return labels;}

			int get_visiableSize() const {return visiableSize;}
			void set_visiableSize(int& v){visiableSize = v;}
			int get_hiddenSize() const{return classesNum;}
			void set_hiddenSize(int& h) { classesNum = h;}
			double get_weight_decay() const {return weightDecay;}
			void set_weight_decay(double& l) {weightDecay = l;}


			double value_gradient(arma::mat& grad);
			void gradient(arma::mat& grad);
			void hessian(arma::mat& grad, arma::mat& hess);
		};
	};
};
#endif