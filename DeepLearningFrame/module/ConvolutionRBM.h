#ifndef CONVOLUTIONRBM_H
#define CONVOLUTIONRBM_H

#include "Module.h"
#include "../util/convolve.h"

namespace dlpft{
	namespace module{
		class ConvolutionRBM : public Module{
			public:
			int inputImageDim;
			int inputImageNum;
			int filterDim;
			int filterNum;
			int outputImageDim;
			int outputImageNum;
		public:
			ConvolutionRBM():Module(){
				name = "ConvolveModule";
			}
			ConvolutionRBM(int input_image_dim,int input_image_num,int filter_dim, int output_num,const ActivationFunction act_func = SIGMOID,const double weightdecay=3e-3){
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
			~ConvolutionRBM(){
			}
			void pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			void CD_k(int k,arma::mat& v, double v_bias);
			void sample_h_given_v(arma::mat& v0_sample, arma::mat& mean, arma::mat& sample);
		    void sample_v_given_h(arma::mat& h, arma::mat& v, arma::mat& sample, double v_bias);
		    arma::mat propup(arma::mat& v);
		    arma::mat propdown(arma::mat& h,double v_bias);
			void  gibbs_hvh(double v_bias,arma::mat& h0_sample);
			arma::mat BiNomial(const arma::mat mean);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
			arma::mat process_delta(arma::mat curr_delta);
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad);
		};
	};
};


#endif
