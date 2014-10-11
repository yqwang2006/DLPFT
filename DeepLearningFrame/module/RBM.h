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
			RBM(int in_size,int out_size,
				const string load_w = "NO",const string w_addr = "",const string b_addr = "",
				const ActivationFunction active_func=SIGMOIDFUNC,const double weightdecay = 3e-3)
				:Module(in_size,out_size,load_w,w_addr,b_addr,active_func,weightdecay){
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
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, double weight_decay,arma::mat& Wgrad, arma::mat& bgrad);
		};
	};
};

#endif