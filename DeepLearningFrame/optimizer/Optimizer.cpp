#include "Optimizer.h"
#include <math.h>
#include <vector>

using namespace dlpft::optimizer;

void dlpft::optimizer::Optimizer::wolfe_line_search(
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
	function_ptr->coefficient = x_new;
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
		

		x_new = x + t*search_dir;
		function_ptr->coefficient = x_new;
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
		x_new = zeros(x.size());
		x_new = x+t*search_dir;
		function_ptr->coefficient = x_new;
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
double dlpft::optimizer::Optimizer::polyinterp(vector<InterPoint>& points, double& xminBound, double& xmaxBound){
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
	//cout << "A:" << endl;
	//cout << A;
	//cout << "b:" << endl;
	//cout << b;


	arma::vec params = arma::solve(A,b);
	//cout << "params:" << endl;
	//cout << params <<endl;

	//cout << "inv_A:" << endl;
	//cout << arma::inv(A);
	//cout << "inv_A*b" << endl;
	//cout << arma::inv(A)*b;

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