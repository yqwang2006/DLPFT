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
			FullConnectModule(int in_size,int out_size):Module(in_size,out_size){
				name = "FullConnection";
				initial_weights_bias();
			}
			FullConnectModule(int in_size,int out_size,ActivationFunction act_func)
				:Module(in_size,out_size,act_func){
				name = "FullConnection";
				initial_weights_bias();
			}
			~FullConnectModule(){}
			void pretrain(const arma::mat data, const arma::imat labels, NewParam param){}
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
		};
	};
};

#endif