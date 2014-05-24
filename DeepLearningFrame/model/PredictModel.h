#ifndef PRED_MODEL_H
#define PRED_MODEL_H
#include "../module/AutoEncoder.h"
#include "../model/ResultModel.h"
#include "../module/SoftMax.h"
#include "../param/AllParam.h"
#include "Model.h"
using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class PredictModel:public Model{
			
		public:
			PredictModel():Model(){}
			void predict(ResultModel* trainModel,arma::mat& testdata, arma::mat& testlabels,vector<NewParam> params);
			double predict_acc(const arma::mat predict_labels, const arma::mat testlabels);
		};
	};
};

#endif