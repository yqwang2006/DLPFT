#ifndef DLMODEL_H
#define DLMODEL_H
#include "Model.h"
using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class TrainModel : public Model{
		private:
			
			
		public:
			TrainModel(){
			}
			ResultModel* pretrain(arma::mat& data,arma::mat& labels, vector<NewParam> param);
			
		};
	};
};

#endif