#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "armadillo"
#include "Module.h"
#include "../function/SoftMaxCost.h"

namespace dlpft{
	namespace module{
		class SoftMax:public Module{
		public:
			arma::mat weightMatrix;
			arma::mat bias;
			SoftMax():Module(){
				name = "SoftMax";
			}
			SoftMax(int in_size,int out_size):Module(in_size,out_size){
				name = "SoftMax";
				initial_params();
			}
			SoftMax(int in_size,int out_size,ActivationFunction act_func)
				:Module(in_size,out_size,act_func){
				name = "SoftMax";
				initial_params();
			}
			~SoftMax(){}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			arma::mat backpropagate( ResultModel& result_model,const arma::mat delta,const arma::mat features,  const arma::imat labels, NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param);
			void initial_params();
			void set_init_coefficient(arma::mat& coefficient);
		};
	};
};

#endif