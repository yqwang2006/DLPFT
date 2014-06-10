#ifndef MODEL_H
#define MODEL_H
#include "armadillo"
#include "../module/AllModule.h"
#include "../param/AllParam.h"

using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class Model{
		protected:
			int layerNumber;
			Module** modules;
			int inputSize;
		public:
			Model(){}
			Model(int input_size,vector<NewParam> module_params){
				inputSize = input_size;
				layerNumber = module_params.size();
				modules = new Module* [layerNumber];
				int in_size = input_size;
				int out_size = 0;
				for(int i = 0; i < layerNumber; i++){
					out_size = atoi(module_params[i].params[params_name[HIDNUM]].c_str());
					modules[i] = create_module(module_params[i],in_size,out_size);
					in_size = out_size;
				}
			}
			~Model(){
				for(int i = 0;i < layerNumber;i++){
					delete modules[i];
					modules[i] = NULL;
				}
				delete []modules;
			}
			Module* create_module(NewParam& param,int in_size,int out_size){
				string m_name = param.params["Algorithm"];

				Module* module;
				if(m_name == "AutoEncoder"){
					module = new AutoEncoder(in_size,out_size);
				}else if(m_name == "RBM"){
					module = new RBM(in_size,out_size);
				}else if(m_name == "SC"){
					module = new SparseCoding(in_size,out_size);
				}else if(m_name == "SoftMax"){
					module = new SoftMax(in_size,out_size);
				}else{
					module = NULL;
				}
				return module;
			}
			Module* create_module(NewParam& param){
				string m_name = param.params["Algorithm"];

				Module* module;
				if(m_name == "ConvolveModule"){
					module = new ConvolveModule();
				}else if(m_name == "Pooling"){
					module = new Pooling();
				}
				return module;
			}
		};
	};
};
#endif