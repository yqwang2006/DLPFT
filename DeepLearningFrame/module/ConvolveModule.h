#ifndef CONVOLVEMODULE_H
#define CONVOLVEMODULE_H
#include "Module.h"
namespace dlpft{
	namespace module{
		
		class ConvolveModule : public Module{
		public:
			int lastOutputDim;
			int lastFilterNum;

		public:
			ConvolveModule():Module(){
				
			}
			ConvolveModule(int last_filter_num,int last_output_dim){
				lastOutputDim = last_output_dim;
				lastFilterNum = last_filter_num;
			}
			~ConvolveModule(){
			}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param){
				ResultModel rm;
				return rm;
			}
			arma::mat backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param);
			
		};

	};
};

#endif