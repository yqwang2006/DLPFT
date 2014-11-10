#ifndef SVMMODULE_H
#define SVMMODULE_H

#include "Module.h"
#include "svm.h"
namespace dlpft{
	namespace module{
		class SvmModule : public Module{
			public:
				svm_model* svmmodel;
				svm_problem prob;
			SvmModule():Module(){
				name = "SvmModule";
			}
			~SvmModule(){
				delete svmmodel;
			}
			void pretrain(const arma::mat data,NewParam param){}
			void train(const arma::mat data, const arma::mat labels, NewParam param);
			arma::mat backpropagate(const arma::mat next_delta, const arma::mat features, NewParam param);
			arma::mat forwardpropagate(const arma::mat data,  NewParam param);
			void initial_weights_bias();
			void set_init_coefficient(arma::mat& coefficient);
			arma::mat process_delta(arma::mat curr_delta)
			{
				arma::mat next_delta = weightMatrix.t()*curr_delta;
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
			void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,double weight_decay, arma::mat& Wgrad, arma::mat& bgrad);

		};

	};
};
#endif