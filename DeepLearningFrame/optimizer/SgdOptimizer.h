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
		public:
			SgdOptimizer(void){ 
				name = "sgd";
				display = true;
				max_iteration = 10;
				tolerance = 1e-9;
				batch_size = 100;
				momentum = 0.95;
				learning_rate = 0.05;

			}
			SgdOptimizer(CostFunction* func,
							int max_iter,
							double learnrate,
							int batch_s,
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

				if(batch_size == 0)
					batch_size = 100;
				if(learning_rate == 0)
					learning_rate = 0.05;
				if(max_iteration == 0)
					max_iteration = 10;

			}
			~SgdOptimizer(void){}
			
			double optimize(string varname);
			double get_tolerance() const{return tolerance;}
			void set_tolerance(double tol){ tolerance = tol;}
		};//class sgdopt
	};//namespace optimizer
};//namespace dlpft
