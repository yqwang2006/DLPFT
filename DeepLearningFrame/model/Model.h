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
			Model(){}
			~Model(){}
			Module* create_module(NewParam& param){
				string m_name = param.params["Algorithm"];

				Module* module;
				if(m_name == "AutoEncoder"){
					module = new AutoEncoder();
				}else if(m_name == "RBM"){
					module = new RBM();
				}else if(m_name == "SC"){
					module = new SparseCoding();
				}else if(m_name == "SoftMax"){
					module = new SoftMax();
				}
				return module;
			}
			Module* create_module(NewParam& param,int last_layer_filter_num,int last_layer_filter_dim){
				string m_name = param.params["Algorithm"];

				Module* module;
				if(m_name == "ConvolveModule"){
					module = new ConvolveModule(last_layer_filter_num,last_layer_filter_dim);
				}else if(m_name == "Pooling"){
					module = new Pooling(last_layer_filter_num,last_layer_filter_dim);
				}
				return module;
			}
		};
	};
};
#endif