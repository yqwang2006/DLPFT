#pragma once
#include "Optimizer.h"
namespace dlpft{
	namespace optimizer{
		class LbfgsOptimizer : public Optimizer
		{
		public:
			LbfgsOptimizer(void){ name = "lbfgs";}
			LbfgsOptimizer(CostFunction* func,int max_iter):Optimizer(func,max_iter){
				name = "lbfgs";
			}
			~LbfgsOptimizer(void);
			double optimize(string );
		};//class sgdopt
	};//namespace optimizer
};//namespace dlpft
