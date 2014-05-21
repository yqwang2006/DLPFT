#ifndef AUTOENCODER_H
#define AUTOENCODER_H
#include "armadillo"
#include "Module.h"
#include "ResultModel.h"
#include "../function/SAECostFunction.h"
#include "../optimizer/CgOptimizer.h"
#include "../optimizer/LbfgsOptimizer.h"
#include "../optimizer/SgdOptimizer.h"
#include "../factory/Creator.h"

namespace dlpft{
	namespace module{
		class AutoEncoder : public Module{
		private:
			
		public:
			AutoEncoder():Module(){}
			~AutoEncoder(){
			}
			ResultModel run(arma::mat& data, arma::mat& labels, NewParam& param);

		};
	};
};

#endif