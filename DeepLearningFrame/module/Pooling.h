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
				inputSize = outputImageDim*outputImageDim*outputImageNum;
				outputSize = inputSize;
			}
			~Pooling(){
			}
			void pretrain(const arma::mat data, const arma::imat labels, NewParam param){}
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
			arma::mat process_delta(arma::mat curr_delta); //up_sampling
		};
	};
};

#endif