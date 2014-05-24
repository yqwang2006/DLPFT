#pragma once
#include "armadillo"
#include <string>
#include "../param/NewParam.h"
#include "ResultModel.h"
using namespace dlpft::param;
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
			virtual void backpropagate(ResultModel& result_model,const arma::mat data, const arma::mat labels, NewParam param)=0;
			virtual arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::mat labels)=0;
		};
	};
};