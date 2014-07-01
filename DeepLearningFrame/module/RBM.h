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
			RBM(int in_size,int out_size,const ActivationFunction active_func=SIGMOID,const double weightdecay = 3e-3)
				:Module(in_size,out_size,active_func,weightdecay){
				name = "RBM";
				initial_weights_bias();
			}
			~RBM();

			void pretrain(const arma::mat data, NewParam param);
			
		    void  CD_k(int k,arma::mat& input_data, arma::mat& v_bias);
			
			void sample_h_given_v(arma::mat& v0_sample, arma::mat& mean, arma::mat& sample);
		    void sample_v_given_h(arma::mat& h, arma::mat& v, arma::mat& sample, arma::mat& v_bias);
		    arma::mat propup(arma::mat& v);
		    arma::mat propdown(arma::mat& h, arma::mat& c_bias);
		    void  gibbs_hvh(arma::mat& v_bias,arma::mat& h0_sample);
		    double get_reconstruct_error(arma::mat& v);
			arma::mat BiNomial(const arma::mat mean);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			void initial_weights_bias();
			arma::mat process_delta(arma::mat curr_delta)
			{
				arma::mat next_delta = zeros(weightMatrix.n_cols,curr_delta.n_cols);
				next_delta = weightMatrix.t()*curr_delta;
				return next_delta;
			}
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad);
		};
	};
};

#endif