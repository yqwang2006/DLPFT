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
			ConvolutionRBM(int input_image_dim,int input_image_num,int filter_dim, int output_num,
				const string load_w = "NO",const string w_addr = "",const string b_addr = "",
				const ActivationFunction act_func = LINEAR,const double weightdecay=3e-3){
				inputImageDim = input_image_dim;
				inputImageNum = input_image_num;
				inputSize = inputImageDim * inputImageDim;
				filterDim = filter_dim;
				outputImageNum = output_num;
				outputImageDim = input_image_dim - filterDim + 1;
				outputSize = outputImageDim*outputImageDim*outputImageNum;
				filterNum = inputImageNum * outputImageNum;
				weightDecay = weightdecay;
				activeFuncChoice = act_func;
				load_weight = load_w;
				weight_addr = w_addr;
				bias_addr = b_addr;
				initial_weights_bias();
			}
			~ConvolutionRBM(){
			}
			bool initial_weights_bias_from_file(string weight_addr,string bias_addr){
				LoadData file(weight_addr);
				LoadData file1(bias_addr);
				clock_t start = clock(),end;
				double dur_time = 0;
				if(!file.load_data(weightMatrix)){
					if(!file.load_data_to_mat(weightMatrix,filterDim*filterNum,filterDim))
						return false;
				}
				if(!file1.load_data(bias)){
					if(!file.load_data_to_mat(bias,outputImageNum,1))
						return false;
				}
				end = clock();
				dur_time = (double)(end-start)/CLOCKS_PER_SEC;
				cout << dur_time << endl;
				return true;
			}
			void pretrain(const arma::mat data,  NewParam param);
			void CD_k(int k,arma::mat& v, double v_bias, mat& h0_mean, mat& h0_samples, mat& nv_means,mat& nv_samples,mat& nh_means,mat& nh_samples);
			void sample_h_given_v(arma::mat& v0_sample, arma::mat& mean, arma::mat& sample);
		    void sample_v_given_h(arma::mat& h, arma::mat& v, arma::mat& sample, double v_bias);
		    arma::mat propup(const arma::mat v);
		    arma::mat propdown(arma::mat& h,double v_bias);
			void  gibbs_hvh(double v_bias,arma::mat& h0_sample, arma::mat& nv_means, arma::mat& nv_samples, arma::mat& nh_means, arma::mat& nh_samples);
			arma::mat BiNomial(const arma::mat mean);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
			arma::mat process_delta(arma::mat curr_delta);
			void crbmGradients(int k,arma::mat minibatch,NewParam param,double v_bias, arma::mat& Wgrad, arma::mat& hgrad, double& vgrad, double& error);
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,double weight_decay, arma::mat& Wgrad, arma::mat& bgrad);
		};
	};
};


#endif
