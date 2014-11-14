#include "CgOptimizer.h"

#include <math.h>
#include <vector>
#include "../util/roots.h"
dlpft::optimizer::CgOptimizer::CgOptimizer(void)
{
	name = "cg";
	max_iteration = 100;
	opt_value = 1e20;
}

dlpft::optimizer::CgOptimizer::~CgOptimizer(void)
{
}

//传入的x是初始位置点
double dlpft::optimizer::CgOptimizer::optimize(string varname){
	
	//计算初始函数值和梯度值
	arma::mat g;
	arma::mat g_old;
	arma::mat search_dir;
	double f = function_ptr->value_gradient(g);
	double f_old;
	int iter = 0;
	double step_size;
	double gtd = 0;
	arma::mat x = function_ptr->coefficient;
	//cout << "before opt:" << function_ptr->get_coefficient()->n_rows<<";" << function_ptr->get_coefficient()->n_cols << endl;
	while(true){
		
		search_direction(g,g_old,iter,search_dir);
		//sim_line_search(f,step_size,x,g,search_dir);
		
		step_size = line_search(f,f_old,iter,x,g,g_old,search_dir);

		if(stop(f,f_old,g,iter)){
			break;
		}
		if(display){
			LogOut << "iteration " << iter << ": ";
			LogOut << "func_value:"<< f << "; step_size = " << step_size << ";" << endl;
			cout << "iteration " << iter << ": ";
			cout << "func_value:"<< f << "; step_size = " << step_size << ";" << endl;
		}
		iter ++;
	}
	//cout << "after opt:" << function_ptr->get_coefficient()->n_rows<<";"  << function_ptr->get_coefficient()->n_cols << endl;
	function_ptr->coefficient = x;
	return f;
}
bool dlpft::optimizer::CgOptimizer::stop(const double& f, const double &f_old, const arma::mat& g, const int& iter){
	if(accu(abs(g)) <= 1e-5)
		return true;
	if(abs(f-f_old) < 1e-9)
		return true;
	if (iter == max_iteration)
		return true;
	return false;
}
void dlpft::optimizer::CgOptimizer::search_direction(
	const arma::mat& grad,
	arma::mat& grad_old,
	const size_t iteration_num, 
	arma::mat& search_dir)
{
	if(iteration_num == 0)
		search_dir = -grad;
	else{
		//double gtgo = arma::dot(grad,grad_old);
		//double gotgo = arma::dot(grad_old, grad_old);
		//cgUpdate == 1
		arma::mat gmgo = (grad-grad_old).t();

		arma::mat betamat = (grad.t()*(grad-grad_old))/(gmgo*search_dir);

		double beta = betamat(0);
		
		search_dir = -grad + beta * search_dir;
	}
	grad_old = grad;
}
double dlpft::optimizer::CgOptimizer::evaluate(arma::mat& x,arma::mat& grad){
	function_ptr->coefficient = x;
	double f_value = function_ptr->value_gradient(grad);
	if(f_value < opt_value){
		opt_location = x;
		opt_value = f_value;
	}
	return f_value;
}
double dlpft::optimizer::CgOptimizer::line_search(
	double& func_value, 
	double& func_value_old, 
	const size_t iter, 
	arma::mat& x,
	arma::mat& grad, 
	arma::mat& grad_old,
	const arma::mat& search_dir)
{
	double gtd = arma::dot(grad,search_dir);
	double t = 0;
	if(iter == 0){
		t = min(1.0,1/accu(abs(grad)));
	}else{
		t = min(1.0,2*(func_value-func_value_old)/gtd);
	}
	if(t <= 0)
		t = 1;
	func_value_old = func_value;
	double gtd_old = gtd;


	double c1 = 1e-4;
	double c2 = 0.2;
	int maxIter = 25;
	double tolX = 1e-9;
	wolfe_line_search(x,t,search_dir,func_value,grad,gtd,c1,c2,maxIter,tolX);

	x = x+t*search_dir;
	return t;
}

