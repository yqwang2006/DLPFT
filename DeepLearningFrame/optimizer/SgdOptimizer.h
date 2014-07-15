#pragma once
#include "Optimizer.h"
namespace dlpft{
	namespace optimizer{
		class SgdOptimizer : public Optimizer
		{
		public:
			double tolerance;
			int batch_size;
			double momentum;
			double learning_rate;
			double learing_rate_decay;
		public:
			SgdOptimizer(void){ 
				name = "sgd";
				display = true;
				max_iteration = 10;
				tolerance = 1e-9;
				batch_size = 100;
				momentum = 0.95;
				learing_rate_decay = 0.98;
				learning_rate = 0.05;

			}
			SgdOptimizer(CostFunction* func,
							int max_iter,
							double learnrate,
							const int batch_s = 100,
							const double learnrate_decay = 0.98,
							const double tol = 1e-9, 
							const double mom = 0.95
							
							)
							:Optimizer(func,max_iter){
				name = "sgd";
				function_ptr = func;
				display = true;
				tolerance = tol;
				batch_size = batch_s;
				momentum = mom;
				learning_rate = learnrate;
				learing_rate_decay = learnrate_decay;
				if(batch_size == 0)
					batch_size = 100;
				if(learning_rate == 0)
					learning_rate = 0.05;
				if(max_iteration == 0)
					max_iteration = 10;
				if(learing_rate_decay == 0)
					learing_rate_decay = 0.98;
			}
			~SgdOptimizer(void){}
			
			double optimize(string varname);
			double get_tolerance() const{return tolerance;}
			void set_tolerance(double tol){ tolerance = tol;}
		};//class sgdopt
	};//namespace optimizer
};//namespace dlpft
