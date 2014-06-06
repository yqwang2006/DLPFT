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
			ResultModel* pretrain(const arma::mat data,const arma::imat labels, vector<NewParam> param);
			void finetune_BP(ResultModel* result_model,const arma::mat data, const arma::imat labels, vector<NewParam> param);
		};
	};
};

#endif