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
		public:
			ConvolveModule():Module(){
				name = "ConvolveModule";
			}
			ConvolveModule(int input_image_dim,
							int input_image_num,
							int filter_dim, 
							int output_num,
							const ActivationFunction act_func = SIGMOID,
							const double weightdecay=3e-3){
				inputImageDim = input_image_dim;
				inputImageNum = input_image_num;
				filterDim = filter_dim;
				outputImageNum = output_num;
				outputImageDim = input_image_dim - filterDim + 1;
				outputSize = outputImageDim*outputImageDim*outputImageNum;
				filterNum = inputImageNum * outputImageNum;
				weightDecay = weightdecay;
				activeFuncChoice = act_func;
				initial_weights_bias();
			}
			~ConvolveModule(){
			}
			void pretrain(const arma::mat data, NewParam param){}
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
			arma::mat process_delta(arma::mat curr_delta);
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad);
		};

	};
};

#endif