#ifndef CONVOLVEMODULE_H
#define CONVOLVEMODULE_H
#include "Module.h"
namespace dlpft{
	namespace module{
		
		class ConvolveModule : public Module{
		public:
			int inputImageDim;
			int inputImageNum;
			int filterDim;
			int filterNum;
			int outputImageDim;
			int outputImageNum;
			arma::mat filters;
			arma::mat bias;
		public:
			ConvolveModule():Module(){
				name = "ConvolveModule";
			}
			ConvolveModule(int input_image_dim,int input_image_num,int filter_dim, int output_num){
				inputImageDim = input_image_dim;
				inputImageNum = input_image_num;
				filterDim = filter_dim;
				outputImageNum = output_num;
				outputImageDim = input_image_dim - filterDim + 1;
				filterNum = inputImageNum * outputImageNum;
				initial_params();
			}
			~ConvolveModule(){
			}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param){
				ResultModel rm;
				return rm;
			}
			arma::mat backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param);
			void initial_params();
		};

	};
};

#endif