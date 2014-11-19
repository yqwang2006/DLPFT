#include "SgdOptimizer.h"
#include "../module/AllModule.h"
double dlpft::optimizer::SgdOptimizer::optimize(string varname){
	arma::mat &x = function_ptr->coefficient;
	arma::mat dat = function_ptr->data;
	arma::mat labels_opt = function_ptr->labels;
	size_t data_dim = size(dat,0);
	size_t data_length = size(dat,1);
	double mom = 0.5;
	int momIncrease = 20;
	arma::mat velocity = zeros(size(x,0),size(x,1));
	int it = 0;
	vector<int> randperm;
	size_t label_dim = size(labels_opt,1);
	arma::mat minibatch(data_dim,batch_size);
	arma::mat batchlabels(batch_size,label_dim);
	arma::mat grad = zeros(size(x,0),size(x,1));
	double func_cost = 0;
	for(int e = 0;e<max_iteration;e++){
		for(int i =0;i < data_length;i++){
			randperm.push_back(i);
		}
		random_shuffle(randperm.begin(),randperm.end());
		double epoch_cost = 0;
		
		
		for(int s = 0;s <= data_length-batch_size;s += batch_size){
			it ++;
			if(it == momIncrease){
				mom = momentum;
			}
			int iter = 0;
			for(int j = 0;j<batch_size;j++,iter++){
				minibatch.col(iter) = dat.col(randperm[s+j]);
				batchlabels.row(iter) = labels_opt.row(randperm[s+j]);
				/*minibatch.col(iter) = dat.col(j);
				batchlabels.row(iter) = labels_opt.row(j);*/
			}

			function_ptr->data = minibatch;
			function_ptr->labels = batchlabels;
			
			func_cost = function_ptr->value_gradient(grad);

			epoch_cost += func_cost;
			

			velocity = mom * velocity - learning_rate*grad;
			x = x+velocity;
			//cout << "Epoch " << e << ": Cost on iteration " << it << " is " << func_cost << endl;
		}
		learning_rate = learing_rate_decay*learning_rate;
		if(display){
			LogOut << "Epoch " << e << ": Cost on iteration is " << epoch_cost << endl;
			cout << "Epoch " << e << ": Cost on iteration is " << epoch_cost << endl;
		}
	}
	return 0;
}