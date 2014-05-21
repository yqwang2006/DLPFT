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
			
			virtual ResultModel run(arma::mat& data, arma::mat& labels, NewParam& param){
				ResultModel rm;
				return rm;
			}
		};
	};
};