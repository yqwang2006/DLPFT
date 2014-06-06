#ifndef FULLCONNECTMODULE_H
#define FULLCONNECTMODULE_H
#include "armadillo"
#include "Module.h"

namespace dlpft{
	namespace module{
		class FullConnectModule:public Module{
		public:
			FullConnectModule():Module(){
				name = "FullConnectModule";
			}
			~FullConnectModule(){}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param){
				ResultModel rm;
				return rm;
			}
			arma::mat backpropagate( ResultModel& result_model,const arma::mat delta,const arma::mat features,  const arma::imat labels, NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param);
		};
	};
};

#endif