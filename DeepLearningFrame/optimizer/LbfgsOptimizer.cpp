#include "LbfgsOptimizer.h"

double dlpft::optimizer::LbfgsOptimizer::optimize(string varname){
	
	//计算初始函数值和梯度值
	arma::mat g;
	arma::mat g_old;
	arma::mat search_dir;
	double f = function_ptr->value_gradient(g);
	double f_old;
	int iter = 0;
	double step_size=1;
	double gtd = 0;
	arma::mat x = function_ptr->coefficient;
	arma::mat old_dirs;
	arma::mat old_stps;
	double Hdiag = 1;
	//cout << "before opt:" << function_ptr->get_coefficient()->n_rows<<";" << function_ptr->get_coefficient()->n_cols << endl;
	while(true){
		
		search_direction(g,g_old,iter,step_size,search_dir,old_dirs,old_stps,Hdiag);
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
bool dlpft::optimizer::LbfgsOptimizer::stop(const double& f, const double &f_old, const arma::mat& g, const int& iter){
	if(sum(sum(abs(g))) <= 1e-5)
		return true;
	if(abs(f-f_old) < 1e-9)
		return true;
	if (iter == max_iteration)
		return true;
	return false;
}
void dlpft::optimizer::LbfgsOptimizer::search_direction(
	const arma::mat& grad,
	arma::mat& grad_old,
	const size_t iteration_num, 
	const double step_size,
	arma::mat& search_dir,
	arma::mat& old_dirs,
	arma::mat& old_stps,
	double& Hdiag)
{
	int corrections = 100;
	if(iteration_num == 0){
		search_dir = -grad;
		old_dirs = zeros(search_dir.n_rows,0);
		old_stps = zeros(search_dir.n_rows,0);
		Hdiag = 1;
	}
	else{
		//double gtgo = arma::dot(grad,grad_old);
		//double gotgo = arma::dot(grad_old, grad_old);
		//cgUpdate == 1
		arma::mat y = grad-grad_old;
		arma::mat s = step_size * search_dir;
		double ys = dot(y,s);
		if(ys > 1e-10){
			int numCorrections = old_dirs.n_cols;
			if(numCorrections < corrections){
				
				old_dirs.insert_cols(numCorrections,s.col(0));
				old_stps.insert_cols(numCorrections,y.col(0));
				
			}else{
				old_dirs.cols(0,corrections-2) = old_dirs.cols(1,corrections-1);
				old_dirs.col(corrections-1) = s.col(0);
				old_stps.cols(0,corrections-2) = old_stps.cols(1,corrections-1);
				old_stps.col(corrections-1) = y.col(0);
			}
			Hdiag = ys/(dot(y,y));
		}
		int p = old_dirs.n_rows;
		int k = old_dirs.n_cols;
		mat ro = zeros(k,1);
		mat q = zeros(p,k+1);
		mat r = zeros(p,k+1);
		mat al = zeros(k,1);
		mat be = zeros(k,1);
		for(int i = 0;i< k;i++){
			ro(i) = ((double)1)/dot(old_stps.col(i),old_dirs.col(i));
		}
		q.col(k) = -grad.col(0);
		for(int i = k-1;i>=0;i--){
			al(i) = ro(i)*dot(old_dirs.col(i),q.col(i+1));
			q.col(i) = q.col(i+1)-al(i)*old_stps.col(i);
		}
		r.col(0) = Hdiag*q.col(0);
		for(int i=0;i<k;i++){
			be(i) = ro(i)*dot(old_stps.col(i).t(),r.col(i));
			r.col(i+1) = r.col(i) + old_dirs.col(i)*(al(i)-be(i));
		}
		search_dir.col(0) = r.col(k);
	}
	grad_old = grad;
}
double dlpft::optimizer::LbfgsOptimizer::line_search(
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
		t = min(1.0,1/sum(sum(abs(grad))));
	}else{
		t = 1;
	}
	if(t <= 0)
		t = 1;
	func_value_old = func_value;
	double gtd_old = gtd;


	double c1 = 1e-4;
	double c2 = 0.9;
	int maxIter = 25;
	double tolX = 1e-9;
	wolfe_line_search(x,t,search_dir,func_value,grad,gtd,c1,c2,maxIter,tolX);

	x = x+t*search_dir;
	return t;
}
