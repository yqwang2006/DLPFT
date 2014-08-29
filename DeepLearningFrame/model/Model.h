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
			string loadWeightFromFile;
			string filePath;
			int inputSize;
		public:
			Model(){}
			Model(int input_size,vector<NewParam> module_params,const string lwff = "NO", const string fp = ""){
				inputSize = input_size;
				layerNumber = module_params.size()-1;
				modules = new Module* [layerNumber];
				int in_size = input_size;
				int in_num = 1;
				loadWeightFromFile = lwff;
				filePath = fp;
				for(int i = 0; i < layerNumber; i++){
					modules[i] = create_module(module_params[i],input_size,in_num,i);
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
			void train_classifier(const arma::mat data, const arma::mat labels, vector<NewParam> param);
			arma::mat predict(const arma::mat testdata, const arma::mat testlabels,vector<NewParam> params);
			Module* create_module(NewParam& param,int& in_size,int& in_num,int layer_id);
			void train(arma::mat data, arma::mat labels,vector<NewParam> model_param);
			void initParams(arma::mat& theta,vector<NewParam> param);
			void modelParamsToStack(arma::mat theta,vector<NewParam> params);
			double predict_acc(const arma::mat predict_labels, const arma::mat labels);
		};
	};
};
#endif