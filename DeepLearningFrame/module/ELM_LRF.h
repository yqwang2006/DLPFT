#ifndef ELMLRF_H
#define ELMLRF_H
#include "Module.h"
namespace dlpft{
	namespace module{
		class ELM_LRF : public Module{
		public:
			mat outputWeight;
			
			ELM_LRF() : Module(){}
			ELM_LRF(int in_size,int out_size,
				const string load_w = "NO",const string w_addr = "",const string b_addr = "",
				const ActivationFunction active_func=SIGMOIDFUNC,const double weightdecay = 3e-3)
				:Module(in_size,out_size,load_w,w_addr,b_addr,active_func,weightdecay){
				name = "DELM";
				initial_weights_bias();
			}
			~ELM_LRF(){
			}
			
			void pretrain(const arma::mat data,NewParam param){}
			void train(const arma::mat data, const arma::mat labels, NewParam param);
			
			mat forwardpropagate(const mat data,  NewParam param);
			void initial_weights_bias();
			bool initial_weights_bias_from_file(string weight_addr,string bias_addr){return false;};
			mat backpropagate(const mat next_delta, const arma::mat features, NewParam param){return zeros(0);};
			
			mat process_delta(mat curr_delta){return zeros(0);};

			void calculate_grad_using_delta(const mat input_data,const mat delta,NewParam param,double weight_decay, mat& Wgrad, mat& bgrad){};

		};
	};
};
#endif