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
			Pooling(int input_image_dim,int input_image_num,int pooling_dim, string pooling_type,const ActivationFunction act = SIGMOID, const double weight = 3e-3){
				inputImageDim = input_image_dim;
				inputImageNum = input_image_num;
				poolingDim = pooling_dim;
				poolingType = pooling_type;
				outputImageDim = inputImageDim / poolingDim;
				outputImageNum = inputImageNum;
				weightDecay = weight;
				activeFuncChoice = act;
				inputSize = inputImageDim*inputImageDim*outputImageNum;
				outputSize = outputImageDim*outputImageDim*outputImageNum;;
				initial_weights_bias();
			}
			~Pooling(){
			}
			void pretrain(const arma::mat data, NewParam param){}
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			void initial_weights_bias();
			arma::mat process_delta(arma::mat curr_delta); //up_sampling
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad);
			arma::mat down_sample(arma::mat data);
		};
	};
};

#endif