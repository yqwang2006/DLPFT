#pragma once
#include <string>
#include "armadillo"
#include <assert.h>
#include "../function/CostFunction.h"
#include "../function/SAECostFunction.h"
using namespace std;
using namespace dlpft::function;
namespace dlpft{
	namespace optimizer{
		class Optimizer{
		public:
			string name;
			CostFunction *function_ptr;
			arma::mat opt_var;
			int max_iteration;
			bool display;
		public:
			Optimizer(void):name(""){
				display = true;
			}
			Optimizer(CostFunction* func,int max_iter):name(""),function_ptr(func),max_iteration(max_iter){
				display = true;
			}
			~Optimizer(){
				
				cout << "~Optimizer()" << endl;
			}

			CostFunction* get_func_ptr(){return function_ptr;}
			void set_func_ptr(CostFunction* ptr){
				function_ptr = ptr;
			}
			void set_opt_var(arma::mat v){
				opt_var = v;
			}
			int get_max_iteration()const{return max_iteration;}
			void set_max_iteration(int mi){max_iteration = mi;}
			virtual double optimize(string varname){return 0;}

		};//class Optimizer
	};//namespace dlpft
};//namespace optimizer
