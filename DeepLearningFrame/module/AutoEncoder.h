#ifndef AUTOENCODER_H
#define AUTOENCODER_H
#include "armadillo"
#include "Module.h"
#include "ResultModel.h"
#include "../function/SAECostFunction.h"
#include "../optimizer/AllOptMethod.h"
#include "../factory/Creator.h"

namespace dlpft{
	namespace module{
		class AutoEncoder : public Module{
		private:
			
		public:
			AutoEncoder():Module(){}
			~AutoEncoder(){
			}
			ResultModel pretrain(const arma::mat data, const arma::mat labels, NewParam param);
			void backpropagate(ResultModel& result_model,const arma::mat data, const arma::mat labels,NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::mat labels);
		};
	};
};

#endif