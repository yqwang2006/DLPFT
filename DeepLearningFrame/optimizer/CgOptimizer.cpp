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
	arma::mat x = function_ptr->get_coefficient();
	//cout << "before opt:" << function_ptr->get_coefficient()->n_rows<<";" << function_ptr->get_coefficient()->n_cols << endl;
	while(true){
		
		search_direction(g,g_old,iter,search_dir);
		//sim_line_search(f,step_size,x,g,search_dir);
		
		step_size = line_search(f,f_old,iter,x,g,g_old,search_dir);

		if(stop(f,f_old,g,iter)){
			break;
		}
		cout << "iteration " << iter << ": ";
		cout << "func_value:"<< f << "; step_size = " << step_size << ";" << endl;
		iter ++;
	}
	//cout << "after opt:" << function_ptr->get_coefficient()->n_rows<<";"  << function_ptr->get_coefficient()->n_cols << endl;
	function_ptr->set_coefficient(x);
	return f;
}
bool dlpft::optimizer::CgOptimizer::stop(const double& f, const double &f_old, const arma::mat& g, const int& iter){
	if(sum(sum(abs(g))) <= 1e-5)
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
	function_ptr->set_coefficient(x);
	double f_value = function_ptr->value_gradient(grad);
	if(f_value < opt_value){
		opt_location = x;
		opt_value = f_value;
	}
	return f_value;
}
//
//bool dlpft::optimizer::CgOptimizer::sim_line_search(double& f_value,
//													double& step_size,
//													arma::mat& x, 
//													arma::mat& grad, 
//													const arma::mat& search_dir){
//	step_size = 1.0;
//	const double inc = 2.1;
//	const double dec = 0.5;
//	double armijo_constant = 1e-4;
//	double wolfe = 0.9;
//	double min_step = 1e-20;
//	double max_step = 1e20;
//	double init_gtd = arma::dot(grad,search_dir);
//	if(init_gtd > 0.0){
//		std::cout << "cg line search dir is not a descent dir" << std::endl;
//		return false;
//	}
//	double init_f_value = f_value;
//	double linear_f_dec = armijo_constant * init_gtd;
//	size_t num_iter = 0;
//	
//	double width = 0;
//	arma::mat new_x;
//
//	while(num_iter < 25){
//		new_x = x;
//		new_x += step_size * search_dir;
//		f_value = evaluate(new_x,grad);
//		if(f_value > init_f_value + step_size * linear_f_dec){
//			width = dec;
//		}else{
//			double gtd = arma::dot(grad,search_dir);
//			if(gtd < wolfe * init_gtd){
//				width = inc;
//			}else{
//				if(gtd > -wolfe * init_gtd){
//					width = dec;
//				}
//				else{
//					break;
//				}
//			}
//		}
//		if(step_size < min_step || step_size > max_step){
//			return false;
//		}
//		//step_size *= width;
//		num_iter ++;
//	}
//	x = new_x;
//	return true;
//}

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
		t = min(1.0,1/sum(sum(abs(grad))));
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

void dlpft::optimizer::CgOptimizer::wolfe_line_search(
	arma::mat &x,
	double& t,
	const arma::mat& search_dir,
	double& func_value,
	arma::mat& grad,
	double& gtd,
	double& c1,
	double& c2,
	int &maxIter,
	double &tolX
	){
	double f_new = 0;
	arma::mat grad_new;
	
	arma::mat x_new = x+t*search_dir;
	function_ptr->set_coefficient(x_new);
	f_new = function_ptr->value_gradient(grad_new);
	double gtd_new = arma::dot(grad_new,search_dir);
	int iter = 0;
	double t_prev = 0;
	double f_prev = func_value;
	arma::mat grad_prev = grad;
	double gtd_prev = gtd;
	double bracket[2],bracketFval[2];
	arma::mat bracketGval[2];
	int done = 0;
	vector<InterPoint> points;
	while(iter < maxIter){
		//第一个if判断的是sufficient decrease条件。如果不满足该条件就跳出循环
		//第二个if判断的是curvature条件，如果进入第二个分支，说明既满足第一个条件又满足第二个条件
		//如果三个分支都不进入，则说明其满足第一个条件和负梯度方向的条件，而不满足curvature条件
		//cout << iter << endl;
		if(f_new > func_value + c1*t*gtd || (iter > 1 && f_new >= f_prev)){
			bracket[0] = t_prev;
			bracket[1] = t;
			bracketFval[0] = f_prev;
			bracketFval[1] = f_new;
			bracketGval[0] = grad_prev;
			bracketGval[1] = grad_new;
			break;
		}else if(abs(gtd_new) <= -c2 * gtd){
			bracket[0] = t;
			bracketFval[0] = f_new;
			bracketGval[0] = grad_new;

			bracket[1] = t;
			bracketFval[1] = f_new;
			bracketGval[1] = grad_new;
			done = 1;
			break;
		}else if(gtd_new >= 0){
			bracket[0] = t_prev;
			bracket[1] = t;
			bracketFval[0] = f_prev;
			bracketFval[1] = f_new;
			bracketGval[0] = grad_prev;
			bracketGval[1] = grad_new;
			break;
		}
		double temp = t_prev;
		t_prev = t;
		double min_step = t + 0.01 * (t - temp);
		double max_step = t * 10;
		InterPoint point1(temp,f_prev,gtd_prev);
		InterPoint point2(t,f_new,gtd_new);
		points.push_back(point1);
		points.push_back(point2);

		t = polyinterp(points,min_step,max_step);

		f_prev = f_new;
		grad_prev = grad_new;
		gtd_prev = gtd_new;
		

		arma::mat x_new = x + t*search_dir;
		function_ptr->set_coefficient(x_new);
		f_new = function_ptr->value_gradient(grad_new);
		gtd_new = arma::dot(grad_new,search_dir);

		iter ++;

	}
	if(iter == maxIter){
		bracket[0] = 0;
		bracket[1] = t;
		bracketFval[0] = func_value;
		bracketFval[1] = f_new;
		bracketGval[0] = grad;
		bracketGval[1] = grad_new;
	}

	int insufProgress = 0;
	int Tpos = 2;
	int LposRemoved = 0;
	double f_Lo;
	int LOpos = 0,HIpos = 0;
	while(!done && iter < maxIter){
		if(bracketFval[0]<bracketFval[1]){
			f_Lo = bracketFval[0];
			LOpos = 0;
			HIpos = 1;
		}else{
			f_Lo = bracketFval[1];
			LOpos = 1;
			HIpos = 0;
		}
		points.clear();
		double g1td = arma::dot(bracketGval[0],search_dir);
		double g2td = arma::dot(bracketGval[1],search_dir);
		InterPoint p1(bracket[0],bracketFval[0],g1td);
		InterPoint p2(bracket[1],bracketFval[1],g2td);
		points.push_back(p1);
		points.push_back(p2);
		double xminBound = 0;
		double xmaxBound = 0;
		double bracket_max = max(bracket[0],bracket[1]) ;
		double bracket_min = min(bracket[0],bracket[1]);
		t = polyinterp(points,xminBound,xmaxBound);
		if(min(bracket_max-t,t-bracket_min) / (bracket_max-bracket_min) < 0.1){
			if(insufProgress || t >= bracket_max|| t <= bracket_min){
				if(abs(t-bracket_max) < abs(t-bracket_min)){
					t = bracket_max - 0.1 * (bracket_max - bracket_min);
				}else{
					t = bracket_min + 0.1 * (bracket_max - bracket_min);
				}
				insufProgress = 0;
			}else{
				insufProgress = 1;
			}
		}else{
			insufProgress = 0;
		}
		arma::mat x_new = zeros(x.size());
		x_new = x+t*search_dir;
		function_ptr->set_coefficient(x_new);
		f_new = function_ptr->value_gradient(grad_new);
		gtd_new = arma::dot(grad_new,search_dir);
		iter ++;

		if(f_new > func_value + c1*t*gtd || f_new >= f_Lo){
			bracket[HIpos] = t;
			bracketFval[HIpos] = f_new;
			bracketGval[HIpos] = grad_new;
			Tpos = HIpos;
		}else{
			if(abs(gtd_new <= -c2*gtd)){
				done = 1;
			}else if(gtd_new * (bracket[HIpos] - bracket[LOpos]) >= 0){
				bracket[HIpos] = bracket[LOpos];
				bracketFval[HIpos] = bracketFval[LOpos];
				bracketGval[HIpos] = bracketGval[LOpos];


				
			}
			bracket[LOpos] = t;
			bracketFval[LOpos] = f_new;
			bracketGval[LOpos] = grad_new;
			Tpos = LOpos;
		}

		if(!done && abs((bracket[0] - bracket[1])*gtd_new) < 1e-9){
				break;
		}


	}
		if(bracketFval[0]<=bracketFval[1]){
			f_Lo = bracketFval[0];
			LOpos = 0;
		}else{
			f_Lo = bracketFval[1];
			LOpos = 1;
		}

		t = bracket[LOpos];
		func_value = bracketFval[LOpos];
		grad = bracketGval[LOpos];
		

}
double dlpft::optimizer::CgOptimizer::polyinterp(vector<InterPoint>& points, double& xminBound, double& xmaxBound){
	int nPoints = points.size();
	//order 是待插值多项式的阶数
	int order = 0;
	double xmin = points[0].x.real;
	double xmax = xmin;
	for(int i = 0;i < nPoints; i++){
		if(points[i].f.isReal())
			order ++;
		if(points[i].g.isReal())
			order ++;
		if(points[i].x.real > xmax)
			xmax = points[i].x.real;
		if(points[i].x.real < xmin)
			xmin = points[i].x.real;
	}
	if(xminBound== 0 && xmaxBound == 0){
		xminBound = xmin;
		xmaxBound = xmax;
	}
	order = order - 1;
	arma::mat A = zeros(nPoints*2,order+1);
	arma::vec b = zeros(nPoints*2);
	arma::mat constraint;
	for(int i = 0;i < nPoints; i++){
		if(points[i].f.isReal()){
			constraint = zeros(1,order+1);
			for(int j = order+1; j > 0;j--){
				constraint(order-j+1) = pow((points[i].x.real),j-1);
			}
			//cout << constraint << endl;
			A.row(i) = constraint;
			b.row(i) = points[i].f.real;
		}
	}
	for(int i = 0;i < nPoints; i++){
		if(points[i].g.isReal()){
			constraint = zeros(1,order+1);
			for(int j = 0; j <order;j++){
				constraint(j) = (order-j)*pow((points[i].x.real),order-j-1);
			}
			//cout << constraint << endl;
			A.row(i+nPoints) = constraint;
			b.row(i+nPoints) = points[i].g.real;
		}
	}
	//求解Ax=b

	arma::vec params = solve(A,b);
	//cout << params <<endl;
	arma::vec dParams = zeros(order,1);
	for(int i = 0;i < order;i++){
		dParams(i) = params(i)*(order-i);
	}
	arma::cx_vec cp;

	if(is_finite(dParams)){
		cp.set_size(2+nPoints);
		cp(0) = xminBound;
		
		cp(1) = xmaxBound;
		for(int i = 0;i < nPoints;i++){
			cp(i+2) = points[i].x.real;
		}
		arma::cx_vec root = roots(dParams);
		//cout << root << endl;
		cp = arma::join_cols(cp,root);
	}else{
		cp.set_size(2+nPoints);
		cp(0) = xminBound;
		cp(1) = xmaxBound;
		for(int i = 0;i < nPoints;i++){
			cp(i+2) = points[i].x.real;
		}
		
	}
	double fmin = datum::inf;
	double minPos = (xminBound + xmaxBound)/2;
	for(int i = 0;i < cp.size();i++){
		std::complex<double> xCp = cp(i);
		std::complex<double> fCp = 0;
		if(xCp.real() >= xminBound && xCp.real() <= xmaxBound){
			fCp = polyval(params,xCp);
			if(fCp.imag() == 0 && fCp.real() < fmin){
				minPos = xCp.real();
				fmin = fCp.real();
			}
		}
	}

	points.clear();
	return minPos;
}