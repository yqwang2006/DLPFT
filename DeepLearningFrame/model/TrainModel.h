#ifndef DLMODEL_H
#define DLMODEL_H
#include "../module/AllModule.h"
#include "../param/AllParam.h"
using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class TrainModel{
		private:
			arma::mat original_data;
			arma::mat original_labels;
			
		public:
			TrainModel(){
			}
			TrainModel(arma::mat d, arma::mat l){
				original_data = d;
				original_labels = l;
			}
			ResultModel* pretrain(arma::mat& data,arma::mat& labels, vector<NewParam> param);
			Module* create_module(NewParam& param);
		};
	};
};

#endif