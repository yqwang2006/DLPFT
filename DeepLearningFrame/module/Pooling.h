#ifndef POOLING_H
#define POOLING_H

#include "Module.h"
namespace dlpft{
	namespace module{
		class Pooling : public Module{
		public:
			int inputImageDim;
			int inputImageNum;
			arma::mat sampleLoc;
			string poolingType;
			int poolingDim;
			int outputImageDim;
			int outputImageNum;
		public:
			Pooling():Module(){
			}
			Pooling(int input_image_dim,int input_image_num,int pooling_dim, string pooling_type){
				inputImageDim = input_image_dim;
				inputImageNum = input_image_num;
				poolingDim = pooling_dim;
				poolingType = pooling_type;
				outputImageDim = inputImageDim / poolingDim;
				outputImageNum = inputImageNum;
			}
			~Pooling(){
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