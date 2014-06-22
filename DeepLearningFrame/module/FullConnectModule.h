#ifndef FULLCONNECTMODULE_H
#define FULLCONNECTMODULE_H
#include "armadillo"
#include "Module.h"

namespace dlpft{
	namespace module{
		class FullConnectModule:public Module{
		public:
			
			FullConnectModule():Module(){
				name = "FullConnection";
			}
			FullConnectModule(int in_size,int out_size,const ActivationFunction active_func=SIGMOID,const double weightdecay = 3e-3)
				:Module(in_size,out_size,active_func,weightdecay){
				name = "FullConnection";
				initial_weights_bias();
			}
			~FullConnectModule(){}
			void pretrain(const arma::mat data, const arma::imat labels, NewParam param){}
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			void initial_weights_bias();
			arma::mat process_delta(arma::mat curr_delta)
			{
				arma::mat next_delta = zeros(weightMatrix.n_cols,curr_delta.n_cols);
				next_delta = weightMatrix.t()*curr_delta;
				return next_delta;
			}
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad);
		};
	};
};

#endif