#ifndef CNN_H
#define CNN_H
#include "Model.h"
using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class CNN : public Model{
			public:
				int resultModelSize;
			CNN(){}
			CNN(int input_size,vector<NewParam> module_params)
				:Model(input_size,module_params)
			{
				layerNumber = module_params.size();
				modules = new Module* [layerNumber];
				int in_size = input_size;
				int in_num = 1;
				resultModelSize = 0;
				for(int i = 0; i < layerNumber; i++){
					modules[i] = create_module(module_params[i],input_size,in_num);
					if(modules[i]->name != "Pooling"){
						resultModelSize ++;
					}
				}
			}
			~CNN(){}
			ResultModel* train(const arma::mat data,const arma::imat labels, vector<NewParam> param);
			void cnnInitParams(arma::mat& theta,vector<NewParam> param);
		};

	};
};


#endif