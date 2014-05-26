#ifndef AUTOENCODER_H
#define AUTOENCODER_H
#include "armadillo"
#include "Module.h"
#include "../function/SAECostFunction.h"
namespace dlpft{
	namespace module{
		class AutoEncoder : public Module{
		private:
			
		public:
			AutoEncoder():Module(){}
			~AutoEncoder(){
			}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			arma::mat backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels);
		};
	};
};

#endif