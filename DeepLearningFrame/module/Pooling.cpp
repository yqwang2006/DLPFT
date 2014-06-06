#include "Pooling.h"
using namespace dlpft::module;

arma::mat Pooling::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	const int samples_num = data.n_cols;
	int image_dim = sqrt(data.n_rows / lastFilterNum);
	pooling_dim = atoi(param.params[params_name[POOLINGDIM]].c_str());
	pooling_type = param.params[params_name[POOLINGTYPE]];
	int result_dim = lastOutputDim / pooling_dim;

	arma::mat pooling_result = arma::zeros(result_dim*result_dim*lastFilterNum,samples_num);
	arma::mat temp_pooling_result = arma::zeros(result_dim,result_dim);
	arma::mat temp_pool_id = arma::zeros(result_dim,result_dim); 
	poolId = arma::zeros(result_dim*result_dim*lastFilterNum,samples_num);
	for(int i = 0;i < samples_num;i++){
		for(int j = 0;j < lastFilterNum;j++){
			arma::mat image = data.col(i).rows(j*lastOutputDim*lastOutputDim,(j+1)*lastOutputDim*lastOutputDim-1);
			image.reshape(lastOutputDim,lastOutputDim);
			for(int poolrow = 0; poolrow < result_dim; poolrow ++){
				int offsetrow = poolrow*pooling_dim;
				for(int poolcol = 0;poolcol < result_dim; poolcol++){
					int offsetcol = poolcol*pooling_dim;
					arma::mat patch = image.submat(offsetrow,offsetcol,offsetrow+pooling_dim-1,offsetcol+pooling_dim-1);
					if(pooling_type == "MEAN"){
						temp_pooling_result(poolrow,poolcol) = arma::sum(arma::sum(patch))/patch.size();
						
					}else if(pooling_type == "STOCHASTIC"){
						arma::mat sto_patch = patch / (arma::sum(arma::sum(patch)));
						int sto_sum = 0;
						double rand_num = std::rand();
						for(int k = 0;k < sto_patch.size();k++){
							if(k == 0){
								if(rand_num > sto_sum && rand_num < sto_patch(k)){
									temp_pooling_result(poolrow,poolcol) = patch(k);
									temp_pool_id(poolrow,poolcol) = k;
									break;
								}
							}else{
								if(rand_num > sto_sum && rand_num < sto_sum + sto_patch(k)){
									temp_pooling_result(poolrow,poolcol) = patch(k);
									temp_pool_id(poolrow,poolcol) = k;
									break;
								}
							}
							sto_sum += sto_patch(k);
							
							sto_patch(k) = sto_sum;
						}
						
					}else{//default max
						temp_pooling_result(poolrow,poolcol) = arma::max(arma::max(patch));
						arma::uvec id = arma::find(patch==arma::max(arma::max(patch)));
						temp_pool_id(poolrow,poolcol) = id(0);
					}

					
				}
			}
			temp_pooling_result.reshape(temp_pooling_result.size(),1);
			temp_pool_id.reshape(temp_pool_id.size(),1);
			pooling_result.col(i).rows(j*result_dim*result_dim,(j+1)*result_dim*result_dim-1) = temp_pooling_result;
			poolId.col(i).rows(j*result_dim*result_dim,(j+1)*result_dim*result_dim-1) = temp_pool_id;
		}


	}

	return pooling_result;
}

arma::mat Pooling::backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param){
	const int samples_num = delta.n_cols;
	int pooled_dim = sqrt(delta.n_rows / lastFilterNum);
	pooling_dim = atoi(param.params[params_name[POOLINGDIM]].c_str());
	pooling_type = param.params[params_name[POOLINGTYPE]];
	int image_dim = pooled_dim + pooling_dim;
	arma::mat curr_delta = arma::zeros(lastOutputDim*lastOutputDim*lastFilterNum,samples_num);

	arma::mat temp_delta = arma::zeros(lastOutputDim,lastOutputDim);
	arma::mat temp_last_delta = arma::zeros(lastOutputDim,lastOutputDim);
	for(int i = 0;i < samples_num;i++){
		for(int j = 0;j < lastFilterNum;j++){
			temp_delta = curr_delta.col(i).rows(j*image_dim*image_dim,(j+1)*image_dim*image_dim-1);
			temp_last_delta = delta.col(i).rows(j*pooled_dim*pooled_dim,(j+1)*pooled_dim*pooled_dim-1);
			temp_delta.reshape(image_dim,image_dim);
			temp_last_delta.reshape(pooled_dim,pooled_dim);
			if(pooling_type == "MEAN"){
				temp_delta = kron(temp_last_delta,ones(pooling_dim));
				temp_delta /= (pooling_dim*pooling_dim);
			}else{
				for(int row = 0;row < temp_last_delta.n_rows; row ++){
					for(int col = 0;col < temp_last_delta.n_cols;col++){
						temp_delta(poolId(row,col)) = temp_last_delta(row,col);
					}
				}


			}
		}
	}
	return curr_delta;

}
