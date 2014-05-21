#ifndef PRED_MODEL_H
#define PRED_MODEL_H
#include "../module/AutoEncoder.h"
#include "../module/ResultModel.h"
#include "../module/SoftMax.h"
#include "../param/AllParam.h"
using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class PredictModel{
		private:
			arma::mat original_data;
			arma::mat original_labels;
			vector<NewParam> params;
			
		public:
			PredictModel(){
			}
			PredictModel( arma::mat d, arma::mat l, vector<NewParam> param){
				original_data = d;
				original_labels = l;
				params = param;
				
			}
			void predict(ResultModel* trainModel,arma::mat& testdata, arma::mat& testlabels,vector<NewParam> params);
			double predict_acc(const arma::mat predict_labels, const arma::mat testlabels);
		};
	};
};

#endif