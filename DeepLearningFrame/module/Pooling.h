#ifndef POOLING_H
#define POOLING_H

#include "Module.h"
namespace dlpft{
	namespace module{
		class Pooling : public Module{
		public:
			int lastOutputDim;
			int lastFilterNum;
			arma::mat poolId;
			bool is_convolve_next;
			string pooling_type;
			int pooling_dim;
		public:
			Pooling():Module(){
				is_convolve_next = true;
			}
			Pooling(int last_filter_num,int last_output_dim){
				lastOutputDim = last_output_dim;
				lastFilterNum = last_filter_num;
				
			}
			~Pooling(){
			}
			ResultModel pretrain(const arma::mat data, const arma::imat labels, NewParam param){
				ResultModel rm;
				return rm;
			}
			arma::mat backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param);
			arma::mat forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param);
			
		};
	};
};

#endif