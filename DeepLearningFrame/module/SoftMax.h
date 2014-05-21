#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "armadillo"
#include "Module.h"
#include "ResultModel.h"
#include "../function/SoftMaxCost.h"
#include "../optimizer/CgOptimizer.h"
#include "../optimizer/LbfgsOptimizer.h"
#include "../optimizer/SgdOptimizer.h"
#include "../factory/Creator.h"

namespace dlpft{
	namespace module{
		class SoftMax:public Module{
		public:
			SoftMax():Module(){
				name = "SoftMax";
			}
			~SoftMax(){}
			ResultModel run(arma::mat& data, arma::mat& labels, NewParam& param);

		};
	};
};

#endif