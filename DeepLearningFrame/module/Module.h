#pragma once
#include "armadillo"
#include <string>
#include "../util/ActiveFunction.h"
#include "../param/NewParam.h"
#include "../model/ResultModel.h"
#include "../param/AllParam.h"
#include "../optimizer/AllOptMethod.h"
#include "../factory/Creator.h"

using namespace dlpft::param;
using namespace dlpft::model;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
namespace dlpft{
	namespace module{
		class Module{
		protected:
			std::string name;
			ActivationFunction active_func_choice;
		public:
			Module(){
				name = "";
				active_func_choice = SIGMOID;
			}
			~Module(){
			}
			
			virtual ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param)=0;
			virtual arma::mat backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels, NewParam param)=0;
			virtual arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param)=0;
		};
	};
};