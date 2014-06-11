#ifndef RBM_H
#define RBM_H
#include "Module.h"
#include "../util/randdata.h"
namespace dlpft{
	namespace module{
		class RBM : public Module{
		public:
			arma::mat h_means;
			arma::mat h_samples;
			arma::mat nv_means;
			arma::mat nv_samples;
			arma::mat nh_means;
			arma::mat nh_samples;

			
			RBM() : Module(){}
			RBM(int in_size,int out_size)
				:Module(in_size,out_size){
				name = "RBM";
				initial_weights_bias();
			}
			RBM(int in_size,int out_size,ActivationFunction act_func)
				:Module(in_size,out_size,act_func){
				name = "RBM";
				initial_weights_bias();
			}
			~RBM();

			void pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			
		    void  CD_k(int k,arma::mat& input_data, arma::mat& weightMat, arma::mat& h_bias, arma::mat& v_bias);
			
			void sample_h_given_v(arma::mat& v0_sample, arma::mat& mean, arma::mat& sample,arma::mat& weightMat, arma::mat& h_bias);
		    void sample_v_given_h(arma::mat& h, arma::mat& v, arma::mat& sample,arma::mat& weightMat, arma::mat& v_bias);
		    arma::mat propup(arma::mat& v,arma::mat& weightMat, arma::mat& h_bias);
		    arma::mat propdown(arma::mat& h,arma::mat& weightMat, arma::mat& c_bias);
		    void  gibbs_hvh(arma::mat& weightMat, arma::mat& h_bias, arma::mat& v_bias,arma::mat& h0_sample);
		    double get_reconstruct_error(arma::mat& v);
			arma::mat BiNomial(const arma::mat mean);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
		};
	};
};

#endif