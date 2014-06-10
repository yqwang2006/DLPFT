#include "SparseCoding.h"
using namespace dlpft::module;
using namespace dlpft::param;
using namespace dlpft::function;
using namespace dlpft::factory;
ResultModel SparseCoding::pretrain(const arma::mat data, const arma::imat labels, NewParam param){
	
	typedef Creator<Optimizer> OptFactory;
	OptFactory& opt_factory = OptFactory::Instance();
	
	int feature_num = atoi(param.params[params_name[HIDNUM]].c_str());
	string opt_method = param.params[params_name[OPTIMETHOD]];
	double learn_rate = atof(param.params[params_name[LEARNRATE]].c_str());
	int max_epoch = atoi(param.params[params_name[MAXEPOCH]].c_str());
	int batch_size = atoi(param.params[params_name[BATCHSIZE]].c_str());
	int visible_size = data.n_rows;
	int samples_num = data.n_cols;
	double lambda = atof(param.params[params_name[SPARSITY]].c_str());
	double gamma = atof(param.params[params_name[GAMMA]].c_str());
	double epsilon = atof(param.params[params_name[EPSILON]].c_str());

	ResultModel result_model;

	
	int donut_dim = sqrt(feature_num);
	
	arma::cube group_cube(feature_num,donut_dim,donut_dim);

	int group_num = 0;
	int pool_dim = 3;
	bool istopo = false;
	arma::mat group_mat  = arma::zeros(feature_num,feature_num);
	if(istopo){
		for( int row = 1; row < donut_dim; row ++){
			for(int col = 1; col < donut_dim; col++){
				for(int i = 0;i < pool_dim; i++)
					for(int j = 0;j < pool_dim;j++)
						group_cube(group_num,i,j) = 1;
				group_num ++;
				cirshift(group_cube,2,-1);
			}
			cirshift(group_cube,1,-1);
		}
		group_cube.reshape(feature_num,feature_num,1);
		group_mat = group_cube.slice(0);
	}else{
		group_mat = arma::eye(feature_num,feature_num);
	}

	SCFeatureCost* sc_cost_func = new SCFeatureCost(visible_size,feature_num,group_mat);
	
	sc_cost_func->weightMatrix = weightMatrix;
	set_init_coefficient(sc_cost_func->coefficient,feature_num,batch_size);
	


	Optimizer *sc_opt = opt_factory.createProduct(param.params[params_name[OPTIMETHOD]]);
	sc_opt->set_max_iteration(20);

	arma::mat minibatch = arma::zeros(visible_size,batch_size);
	rand_data(data,minibatch,samples_num,batch_size);
	
	
	int num_batches = ceil((double)samples_num/batch_size);
	
	double error = 0;
	arma::mat error_mat = arma::zeros(feature_num,batch_size);
	//begin iteration:
	for(int iter = 0;iter < max_epoch; iter++){
		for(int batch = 0;batch < num_batches;batch++){
			error_mat = sc_cost_func->weightMatrix.t() * sc_cost_func->coefficient - minibatch;
			
			error = arma::sum(arma::sum(arma::pow(error_mat,2)))/batch_size;
			//cout << "error = " << error << endl;
			double fResidue = error;
			arma::mat R = group_mat * arma::pow(sc_cost_func->coefficient,2);
			R = arma::sqrt(R+epsilon);
			double Jsparsity = lambda*arma::sum(arma::sum(R));
			double Jweight = gamma * arma::sum(arma::sum(arma::pow(sc_cost_func->weightMatrix,2)));

			cout << iter << "	" << batch << "	" << fResidue+Jsparsity+Jweight << "	" << fResidue
				<< "	"<< Jsparsity << "	" << Jweight << endl;

			
			rand_data(data,minibatch,samples_num,batch_size);
			
			
			sc_cost_func->coefficient = sc_cost_func->weightMatrix * minibatch;
			arma::vec normWM = arma::sum(arma::pow(sc_cost_func->weightMatrix,2),1);

			for(int i = 0;i < sc_cost_func->coefficient.n_cols;i++){
				sc_cost_func->coefficient.col(i) /= normWM;
			}
			

			sc_cost_func->data = minibatch;
			sc_cost_func->coefficient.reshape(feature_num*batch_size,1);

			sc_opt->set_func_ptr(sc_cost_func);
			
			
			//begin optimize
			sc_opt->optimize("coefficient");
			//end optimize
			sc_cost_func->coefficient.reshape(feature_num,batch_size);

			sc_cost_func->weightMatrix = ((minibatch*sc_cost_func->coefficient.t())
				*arma::inv((gamma*batch_size*eye(feature_num,feature_num)+sc_cost_func->coefficient*sc_cost_func->coefficient.t()) )).t();


		}
	}
	result_model.weightMatrix = sc_cost_func->weightMatrix;

	return result_model;
}
void SparseCoding::rand_data(const arma::mat input, arma::mat& batch,int sample_num, int batch_size){
	
	srand(unsigned(time(NULL)));
	int batches_num = sample_num / batch_size;
	int visible_size = input.n_rows;
	vector<int> groups;
	for(int i = 0;i < sample_num; i++){
			groups.push_back(i);
	}

	random_shuffle(groups.begin(),groups.end());

	for(int i = 0;i < batch_size; i++)
	{
		batch.col(i) = input.col(groups[i]);
	}

	


}
void SparseCoding::cirshift(arma::cube& group_cube,int dim, int dir){
	arma::cube temp_cube = group_cube;
	if(dim == 0){
		//group_cube.tube
		for(int i = 0;i < group_cube.n_slices;i++){
			for(int j = 0;j < group_cube.n_rows;j++){
				int swap_j = (j+dir);
				if(swap_j < 0)
					swap_j += group_cube.n_rows;
				else if(swap_j >= group_cube.n_rows)
					swap_j -= group_cube.n_rows;
				group_cube.slice(i).row(swap_j) = temp_cube.slice(i).row(j);
			}
		}
		
	}else if(dim == 1){
		//group_cube.tube
		for(int i = 0;i < group_cube.n_slices;i++){
			for(int j = 0;j < group_cube.n_cols;j++){
				int swap_j = (j+dir);
				if(swap_j < 0)
					swap_j += group_cube.n_cols;
				else if(swap_j >= group_cube.n_cols)
					swap_j -= group_cube.n_cols;
				group_cube.slice(i).col(swap_j) = temp_cube.slice(i).col(j);
			}
		}
		
	}else if(dim == 2){
		for(int j = 0;j < group_cube.n_slices;j++){
			int swap_j = (j+dir);
			if(swap_j < 0)
				swap_j += group_cube.n_slices;
			else if(swap_j >= group_cube.n_slices)
				swap_j -= group_cube.n_slices;
			group_cube.slice(swap_j) = temp_cube.slice(j);
		}
	}

}
arma::mat SparseCoding::backpropagate( ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels, NewParam param){
	arma::mat curr_delta;
	curr_delta.fill(1.0);
	return curr_delta;
	
}
arma::mat SparseCoding::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	arma::mat features = result_model.weightMatrix * data;
	return features;

}
void SparseCoding::initial_params(){

	weightMatrix = arma::randu<arma::mat> (outputSize,inputSize);
	featureMatrix = arma::randu<arma::mat> (inputSize,outputSize);

}
void SparseCoding::set_init_coefficient(arma::mat& coefficient,int rows, int cols){
	coefficient = arma::randu<arma::mat> (rows,cols);
}