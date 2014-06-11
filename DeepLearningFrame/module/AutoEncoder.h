#ifndef AUTOENCODER_H
#define AUTOENCODER_H
#include "armadillo"
#include "Module.h"
#include "../function/SAECostFunction.h"
namespace dlpft{
	namespace module{
		class AutoEncoder : public Module{
		public:
			arma::mat backwardWeight;
			arma::mat backwardBias;

			AutoEncoder():Module(){
				name = "AutoEncoder";
			}
			AutoEncoder(int in_size,int out_size)
				:Module(in_size,out_size){
				name = "AutoEncoder";
				initial_weights_bias();
			}
			AutoEncoder(int in_size,int out_size,ActivationFunction act_func)
				:Module(in_size,out_size,act_func){
				name = "AutoEncoder";
				initial_weights_bias();
			}
			~AutoEncoder(){
			}
			void pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
			void set_init_coefficient(arma::mat& coefficeint);
		};
	};
};

#endif