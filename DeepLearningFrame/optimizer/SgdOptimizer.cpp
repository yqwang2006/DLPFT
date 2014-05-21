#include "SgdOptimizer.h"

double dlpft::optimizer::SgdOptimizer::optimize(string varname){
	arma::mat x = function_ptr->get_coefficient();
	arma::mat dat = function_ptr->get_data();
	size_t data_dim = size(dat,0);
	size_t data_length = size(dat,1);
	double mom = 0.5;
	int momIncrease = 20;
	arma::mat velocity = zeros(size(x,0),size(x,1));
	int it = 0;
	vector<int> randperm;
	arma::mat minibatch(data_dim,batch_size);
	arma::mat grad = zeros(size(x,0),size(x,1));
	double func_cost = 0;
	for(int e = 0;e<max_iteration;e++){
		for(int i =0;i < data_length;i++){
			randperm.push_back(i);
		}
		random_shuffle(randperm.begin(),randperm.end());
		
		
		
		for(int s = 0;s <= data_length-batch_size;s += batch_size){
			it ++;
			if(it == momIncrease){
				mom = momentum;
			}
			int iter = 0;
			for(int j = 0;j<batch_size;j++,iter++){
				minibatch.col(iter) = dat.col(randperm[s+j]);
				//minibatch.col(iter) = dat.col(s+j);
			}
			


			function_ptr->set_data(minibatch);
			function_ptr->set_coefficient(x);
			
			func_cost = function_ptr->value_gradient(grad);

			velocity = mom * velocity + alpha*grad;
			x = function_ptr->get_coefficient()-velocity;
			cout << "Epoch " << e << ": Cost on iteration " << it << " is " << func_cost << endl;
		}
		alpha = alpha/2;
	}
	function_ptr->set_coefficient(x);
	return 0;
}