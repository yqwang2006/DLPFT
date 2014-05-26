#ifndef DLMODEL_H
#define DLMODEL_H
#include "Model.h"
using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class TrainModel : public Model{
		private:
			bool finetune_switch;	
			
		public:
			TrainModel(){
				finetune_switch = true;
			}
			ResultModel* pretrain(arma::mat& data,arma::imat& labels, vector<NewParam> param);
			
		};
	};
};

#endif