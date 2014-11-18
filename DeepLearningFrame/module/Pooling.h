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
			Pooling(int input_image_dim,int input_image_num,int pooling_dim, string pooling_type,
				const string load_w = "NO",const string w_addr = "",const string b_addr = "",
				const ActivationFunction act = SIGMOIDFUNC, const double weight = 3e-3){
				name = "Pooling";
				inputImageDim = input_image_dim;
				inputImageNum = input_image_num;
				poolingDim = pooling_dim;
				poolingType = pooling_type;
				outputImageDim = inputImageDim / poolingDim;
				outputImageNum = inputImageNum;
				weightDecay = weight;
				activeFuncChoice = act;
				inputSize = inputImageDim*inputImageDim*outputImageNum;
				outputSize = outputImageDim*outputImageDim*outputImageNum;
				load_weight = load_w;
				weight_addr = w_addr;
				bias_addr = b_addr;
				initial_weights_bias();
			}
			~Pooling(){
			}
			void pretrain(const arma::mat data, NewParam param){}
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			arma::mat backpropagate(const arma::mat next_delta, const arma::mat features, NewParam param);
			void initial_weights_bias();
			arma::mat process_delta(arma::mat curr_delta); //up_sampling
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,double weight_decay, arma::mat& Wgrad, arma::mat& bgrad);
			arma::mat down_sample(arma::mat data);
			bool initial_weights_bias_from_file(string weight_addr,string bias_addr){
				LoadData file(weight_addr);
				LoadData file1(bias_addr);
				clock_t start = clock(),end;
				double dur_time = 0;
				if(!file.load_data(weightMatrix)){
					if(!file.load_data_to_mat(weightMatrix,outputSize,inputSize))
						return false;
				}
				if(!file1.load_data(bias)){
					if(!file.load_data_to_mat(bias,outputSize,1))
						return false;
				}
				end = clock();
				dur_time = (double)(end-start)/CLOCKS_PER_SEC;
				cout << dur_time << endl;
				return true;
			}
		};
	};
};

#endif