#pragma once
#include "armadillo"
#include <string>
#include "../util/ActiveFunction.h"
#include "../param/NewParam.h"
#include "../param/AllParam.h"
#include "../optimizer/AllOptMethod.h"
#include "../factory/Creator.h"
#include "../util/params_name.h"
#include "../util/global_vars.h"
#include "../io/LoadData.h"
//#define DEBUG 1
using namespace dlpft::param;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
using namespace dlpft::io;
namespace dlpft{
	namespace module{
		class Module{
		public:
			std::string name;
			ActivationFunction activeFuncChoice;
			int inputSize;
			int outputSize;
			double weightDecay;
			std::string load_weight;
			arma::mat weightMatrix;
			arma::mat bias;
			std::string weight_addr;
			std::string bias_addr;
		public:
			Module(){
				name = "";
				activeFuncChoice = SIGMOID;
			}
			Module(int in_size,int out_size,
				const string load_w = "NO",const string w_addr = "",const string b_addr = "",
				const ActivationFunction active_func=SIGMOID,const double weightdecay = 3e-3)
				:inputSize(in_size),outputSize(out_size){
				inputSize = in_size;
				outputSize = out_size;
				name = "";
				weight_addr = w_addr;
				bias_addr = b_addr;
				weightDecay = weightdecay;
				activeFuncChoice = active_func;
				load_weight = load_w;
			}
			~Module(){
			}
			
			virtual void pretrain(const arma::mat data, NewParam param)=0;
			
			virtual arma::mat forwardpropagate(const arma::mat data,  NewParam param)=0;
			virtual void initial_weights_bias() = 0;
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
			virtual arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param)=0;
			
			virtual arma::mat process_delta(arma::mat curr_delta) = 0;

			virtual void calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,double weight_decay, arma::mat& Wgrad, arma::mat& bgrad)=0;
		};
	};
};