#ifndef MODEL_H
#define MODEL_H
#include "armadillo"
#include "../module/AllModule.h"
#include "../param/AllParam.h"


using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class Model{
		public:
			int layerNumber;
			Module** modules;
			int inputSize;
		public:
			Model(){}
			Model(int input_size,vector<NewParam> module_params){
				inputSize = input_size;
				layerNumber = module_params.size()-1;
				modules = new Module* [layerNumber];
				int in_size = input_size;
				int in_num = 1;
				for(int i = 0; i < layerNumber; i++){
					modules[i] = create_module(module_params[i],input_size,in_num);
				}
			}
			~Model(){
				for(int i = 0;i < layerNumber;i++){
					delete modules[i];
					modules[i] = NULL;
				}
				delete []modules;
			}
			void pretrain(arma::mat data, vector<NewParam> model_param);
			void train_classifier(const arma::mat data, const arma::imat labels, vector<NewParam> param);
			arma::imat predict(const arma::mat testdata, const arma::imat testlabels,vector<NewParam> params);
			Module* create_module(NewParam& param,int& in_size,int& in_num);
			void train(arma::mat data, arma::imat labels,vector<NewParam> model_param);
			void initParams(arma::mat& theta,vector<NewParam> param);
			void modelParamsToStack(arma::mat theta,vector<NewParam> params);
			double predict_acc(const arma::imat predict_labels, const arma::imat labels);
		};
	};
};
#endif