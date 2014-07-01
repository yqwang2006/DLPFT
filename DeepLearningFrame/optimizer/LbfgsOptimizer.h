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
			double line_search(double& func_value, double& func_value_old, const size_t iter, arma::mat& x,arma::mat& grad,arma::mat& grad_old, const arma::mat& search_dir);
			void search_direction(const arma::mat& grad,arma::mat& grad_old,const size_t iteration_num, const double step_size,arma::mat& search_dir,arma::mat& old_dirs, arma::mat& old_stps,double& Hdiag);
			//bool sim_line_search(double& f_value,double& step_size, arma::mat& x, arma::mat& grad, const arma::mat& search_dir);
			double evaluate(arma::mat& x, arma::mat& grad);
			bool stop(const double& f, const double &f_old, const arma::mat& g, const int& iter);
		};//class sgdopt
	};//namespace optimizer
};//namespace dlpft
