#pragma once
#include "armadillo"
#include <string>
#include "../util/ActiveFunction.h"
#include "../param/NewParam.h"
#include "../model/ResultModel.h"
#include "../param/AllParam.h"
#include "../optimizer/AllOptMethod.h"
#include "../factory/Creator.h"
#include "../util/params_name.h"
using namespace dlpft::param;
using namespace dlpft::model;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
namespace dlpft{
	namespace module{
		class Module{
		public:
			std::string name;
			ActivationFunction activeFuncChoice;
			int inputSize;
			int outputSize;
			arma::mat weightMatrix;
			arma::mat bias;
		public:
			Module(){
				name = "";
				activeFuncChoice = SIGMOID;
			}
			Module(int in_size,int out_size):inputSize(in_size),outputSize(out_size){
				name = "";
				activeFuncChoice = SIGMOID;
			}
			Module(int in_size,int out_size,ActivationFunction active_func):inputSize(in_size),outputSize(out_size){
				name = "";
				activeFuncChoice = active_func;
			}
			~Module(){
			}
			
			virtual void pretrain(const arma::mat data, const arma::imat labels, NewParam param)=0;
			virtual arma::mat backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param)=0;
			virtual arma::mat forwardpropagate(const arma::mat data,  NewParam param)=0;
			virtual void initial_weights_bias() = 0;
			arma::mat process_delta(arma::mat curr_delta){
				return curr_delta;
			}
		};
	};
};