#pragma once
#include "Optimizer.h"
#include "../util/InterPoint.h"
using namespace dlpft::function;
namespace dlpft{
	namespace optimizer{
		class CgOptimizer :
			public Optimizer
		{
		public:
			CgOptimizer(void);
			
			CgOptimizer(const int);
			CgOptimizer(CostFunction* func,int max_iter):Optimizer(func,max_iter){
				name = "cg";
			}
			~CgOptimizer(void);
		
			double optimize(string varname);
			
			void search_direction(const arma::mat& grad,arma::mat& grad_old,const size_t iteration_num, arma::mat& search_dir);
			double line_search(double& func_value, double& func_value_old, const size_t iter, arma::mat& x,arma::mat& grad,arma::mat& grad_old, const arma::mat& search_dir);
			void wolfe_line_search(arma::mat &x,double& t,const arma::mat& search_dir,double& func_value,arma::mat& grad,double& gtd,double& c1,double& c2,int &maxIter,double &tolX);
			double polyinterp(vector<InterPoint>&, double&, double&);
			//bool sim_line_search(double& f_value,double& step_size, arma::mat& x, arma::mat& grad, const arma::mat& search_dir);
			double evaluate(arma::mat& x, arma::mat& grad);
			bool stop(const double& f, const double &f_old, const arma::mat& g, const int& iter);
			
		private:
			
			arma::mat opt_location;
			double opt_value;
		};
	};
};
