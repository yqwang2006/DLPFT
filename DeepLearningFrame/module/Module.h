#pragma once
#include "armadillo"
#include <string>
#include "../param/NewParam.h"
#include "../model/ResultModel.h"

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
		public:
			Module(){
			}
			~Module(){
			}
			
			virtual ResultModel pretrain(const arma::mat data, const arma::mat labels, NewParam param)=0;
			virtual arma::mat backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::mat labels, NewParam param)=0;
			virtual arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::mat labels)=0;
		};
	};
};