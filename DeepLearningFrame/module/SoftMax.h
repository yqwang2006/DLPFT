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
			SoftMax(int in_size,int out_size):Module(in_size,out_size){
				name = "SoftMax";
				activeFuncChoice = SOFTMAX;
				initial_weights_bias();
			}
			SoftMax(int in_size,int out_size,ActivationFunction act_func)
				:Module(in_size,out_size,act_func){
				name = "SoftMax";
				activeFuncChoice = SOFTMAX;
				initial_weights_bias();
			}
			~SoftMax(){}
			void pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
			void set_init_coefficient(arma::mat& coefficient);
			arma::mat process_delta(arma::mat curr_delta)
			{
				arma::mat next_delta = zeros(weightMatrix.n_cols,curr_delta.n_cols);
				next_delta = weightMatrix.t()*curr_delta;
				return next_delta;
			}
		};
	};
};

#endif