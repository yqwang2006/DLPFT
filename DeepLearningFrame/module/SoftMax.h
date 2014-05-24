#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "armadillo"
#include "Module.h"
#include "../function/SoftMaxCost.h"

namespace dlpft{
	namespace module{
		class SoftMax:public Module{
		public:
			SoftMax():Module(){
				name = "SoftMax";
			}
			~SoftMax(){}
			ResultModel pretrain(const arma::mat data, const arma::mat labels, NewParam param);
			arma::mat backpropagate( ResultModel& result_model,const arma::mat delta,const arma::mat features,  const arma::mat labels, NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::mat labels);
		};
	};
};

#endif