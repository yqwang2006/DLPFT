#ifndef SPARSECODING_H
#define SPARSECODING_H

#include "Module.h"
#include "../function/SCFeatureCost.h"
#include "../function/ScWeightFunction.h"
namespace dlpft{
	namespace module{
		class SparseCoding : public Module{
		public:
			arma::mat weightMatrix;
			arma::mat featureMatrix;
			SparseCoding():Module(){name = "Sparse Coding";}
			SparseCoding(int in_size,int out_size):Module(in_size,out_size){
				name = "Sparse Coding";
				initial_params();
			}
			~SparseCoding(){}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param);
			arma::mat backpropagate( ResultModel& result_model,const arma::mat delta,const arma::mat features, const arma::imat labels, NewParam param);
			void cirshift(arma::cube& group_cube,int dim, int dir);
			void rand_data(const arma::mat input, arma::mat& batch,int sample_num, int batch_size);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param);
			void initial_params();
			void set_init_coefficient(arma::mat& coefficeint,int rows, int cols);
		};
		
	};
};


#endif