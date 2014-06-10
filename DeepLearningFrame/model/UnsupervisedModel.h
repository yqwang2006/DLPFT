#ifndef UnsupervisedModel_H
#define UnsupervisedModel_H
#include "Model.h"
using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class UnsupervisedModel : public Model{
		private:
			bool finetune_switch;	
			
		public:
			UnsupervisedModel(){
				finetune_switch = false;
			}
			UnsupervisedModel(int input_dim,vector<NewParam> module_params)
				:Model(input_dim,module_params)
			{
				finetune_switch = false;
			}
			ResultModel* pretrain(const arma::mat data,const arma::imat labels, vector<NewParam> param);
			void finetune_BP(ResultModel* result_model,const arma::mat data, const arma::imat labels, vector<NewParam> param);
			
			void predict(ResultModel* UnsupervisedModel,arma::mat& testdata, arma::imat& testlabels,vector<NewParam> params);
			double predict_acc(const arma::imat predict_labels, const arma::imat testlabels);

		};
	};
};

#endif