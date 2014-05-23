#pragma once
#include "Optimizer.h"
namespace dlpft{
	namespace optimizer{
		class LbfgsOptimizer : public Optimizer
		{
		public:
			LbfgsOptimizer(void){ name = "lbfgs";cout << name << endl;}
			~LbfgsOptimizer(void);
			double optimize(string );
		};//class sgdopt
	};//namespace optimizer
};//namespace dlpft
