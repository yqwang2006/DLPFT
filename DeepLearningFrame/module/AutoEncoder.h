#ifndef AUTOENCODER_H
#define AUTOENCODER_H
#include "armadillo"
#include "Module.h"
#include "../function/SAECostFunction.h"
namespace dlpft{
	namespace module{
		class AutoEncoder : public Module{
		public:
			arma::mat forwardWeight;
			arma::mat forwardBias;
			arma::mat backwardWeight;
			arma::mat backwardBias;

			AutoEncoder():Module(){
				name = "AutoEncoder";
			}
			AutoEncoder(int in_size,int out_size)
				:Module(in_size,out_size){
				name = "AutoEncoder";
				initial_params();
			}
			AutoEncoder(int in_size,int out_size,ActivationFunction act_func)
				:Module(in_size,out_size,act_func){
				name = "AutoEncoder";
				initial_params();
			}
			~AutoEncoder(){
			}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			arma::mat backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param);
			void initial_params();
			void set_init_coefficient(arma::mat& coefficeint);
		};
	};
};

#endif