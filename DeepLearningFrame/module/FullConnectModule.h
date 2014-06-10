#ifndef FULLCONNECTMODULE_H
#define FULLCONNECTMODULE_H
#include "armadillo"
#include "Module.h"

namespace dlpft{
	namespace module{
		class FullConnectModule:public Module{
		public:
			
			arma::mat weightMatrix;
			arma::mat bias;
			FullConnectModule():Module(){
				name = "FullConnectModule";
			}
			FullConnectModule(int in_size,int out_size):Module(in_size,out_size){
				name = "FullConnectModule";
				initial_params();
			}
			FullConnectModule(int in_size,int out_size,ActivationFunction act_func)
				:Module(in_size,out_size,act_func){
				name = "FullConnectModule";
				initial_params();
			}
			~FullConnectModule(){}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param){
				ResultModel rm;
				return rm;
			}
			arma::mat backpropagate( ResultModel& result_model,const arma::mat delta,const arma::mat features,  const arma::imat labels, NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param);
			void initial_params();
		};
	};
};

#endif