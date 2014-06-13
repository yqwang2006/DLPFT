#ifndef SPARSECODING_H
#define SPARSECODING_H

#include "Module.h"
#include "../function/SCFeatureCost.h"
#include "../function/ScWeightFunction.h"
namespace dlpft{
	namespace module{
		class SparseCoding : public Module{
		public:
			SparseCoding():Module(){name = "Sparse Coding";}
			SparseCoding(int in_size,int out_size):Module(in_size,out_size){
				name = "Sparse Coding";
				initial_weights_bias();
			}
			~SparseCoding(){}
			void pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param);
			void cirshift(arma::cube& group_cube,int dim, int dir);
			void rand_data(const arma::mat input, arma::mat& batch,int sample_num, int batch_size);
			arma::mat forwardpropagate(const arma::mat data, NewParam param);
			void initial_weights_bias();
			void set_init_coefficient(arma::mat& coefficeint,int rows, int cols);
			arma::mat process_delta(arma::mat curr_delta)
			{
				return weightMatrix.t()*curr_delta;
			}
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param, arma::mat& Wgrad, arma::mat& bgrad);
		};
		
	};
};


#endif