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
			double alpha;
		public:
			SgdOptimizer(void){ 
				name = "sgd";
				max_iteration = 3;
				tolerance = 1e-9;
				batch_size = 50;
				momentum = 0.95;
				alpha = 0.05;

			}
			SgdOptimizer(CostFunction* func,
							double& s_size,  
							double& alp,
							int& bs,
							const double tol = 1e-9, 
							const double mom = 0.95, 
							const int max_iter = 3
							)
				:tolerance(tol){
				max_iteration = max_iter;
				name = "sgd";
				function_ptr = func;
				tolerance = tol;
				batch_size = bs;
				momentum = mom;
				alpha = alp;
			}
			~SgdOptimizer(void){cout << "~SgdOptimizer()" << endl;}
			
			double optimize(string varname);
			double get_tolerance() const{return tolerance;}
			void set_tolerance(double tol){ tolerance = tol;}
		};//class sgdopt
	};//namespace optimizer
};//namespace dlpft
