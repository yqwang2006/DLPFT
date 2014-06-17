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
				layerNumber = module_params.size();
			}
			~Model(){
				for(int i = 0;i < layerNumber;i++){
					delete modules[i];
					modules[i] = NULL;
				}
				delete []modules;
			}
			Module* create_module(NewParam& param,int& in_size,int& in_num){
				string m_name = param.params["Algorithm"];
				int out_size = atoi(param.params[params_name[HIDNUM]].c_str());
				Module* module;
				if(m_name == "AutoEncoder"){
					module = new AutoEncoder(in_size,out_size);
					in_size = out_size;
				}else if(m_name == "RBM"){
					module = new RBM(in_size,out_size);
					in_size = out_size;
				}else if(m_name == "SC"){
					module = new SparseCoding(in_size,out_size);
					in_size = out_size;
				}else if(m_name == "SoftMax"){
					module = new SoftMax(in_size,out_size);
					in_size = out_size;
				}else if(m_name == "ConvolveModule"){
					int in_dim = sqrt(in_size / in_num);
					int filter_dim = atoi(param.params[params_name[FILTERDIM]].c_str());
					int out_num = atoi(param.params[params_name[FILTERNUM]].c_str());
					module = new ConvolveModule(in_dim,in_num,filter_dim,out_num);
					int out_dim = in_dim - filter_dim + 1;
					in_size = out_dim*out_dim*out_num;
					in_num = out_num;
				}else if(m_name == "CRBM"){
					int in_dim = sqrt(in_size / in_num);
					int filter_dim = atoi(param.params[params_name[FILTERDIM]].c_str());
					int out_num = atoi(param.params[params_name[FILTERNUM]].c_str());
					module = new ConvolutionRBM(in_dim,in_num,filter_dim,out_num);
					int out_dim = in_dim - filter_dim + 1;
					in_size = out_dim*out_dim*out_num;
					in_num = out_num;
				}else if(m_name == "Pooling"){
					int in_dim = sqrt(in_size/in_num);
					int pool_dim = atoi(param.params[params_name[POOLINGDIM]].c_str());
					string pool_type = param.params[params_name[POOLINGTYPE]];
					module = new Pooling(in_dim,in_num,pool_dim,pool_type);
					int out_dim = in_dim/pool_dim;
					in_size = out_dim * out_dim * in_num;
					in_num = in_num;
				}else if(m_name == "FullConnection"){
					int o_size = atoi(param.params[params_name[HIDNUM]].c_str());
					module = new FullConnectModule(in_size,o_size);
					in_size = o_size;
				}else{
					module = NULL;
				}
				return module;
			}
		};
	};
};
#endif