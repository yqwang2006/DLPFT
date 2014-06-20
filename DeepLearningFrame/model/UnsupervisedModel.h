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
				finetune_switch = true;
			}
			UnsupervisedModel(int input_size,vector<NewParam> module_params)
				:Model(input_size,module_params)
			{
				finetune_switch = false;
				layerNumber = module_params.size();
				modules = new Module* [layerNumber];
				int in_size = input_size;
				int input_maps = 1;
				for(int i = 0; i < layerNumber; i++){
					modules[i] = create_module(module_params[i],in_size,input_maps);
				}
			}
			void pretrain(const arma::mat data,const arma::imat labels, vector<NewParam> param);
			void finetune_BP(const arma::mat data, const arma::imat labels, vector<NewParam> param);
			
			double predict(const arma::mat testdata, const arma::imat testlabels,vector<NewParam> params,arma::imat& pred_labels );
			double predict_acc(const arma::imat predict_labels, const arma::imat testlabels);

		};
	};
};

#endif