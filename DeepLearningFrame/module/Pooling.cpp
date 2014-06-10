#include "Pooling.h"
using namespace dlpft::module;

arma::mat Pooling::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	const int samples_num = data.n_cols;
	

	arma::mat pooling_result = arma::zeros(outputImageDim*outputImageDim*inputImageNum,samples_num);
	arma::mat temp_pooling_result = arma::zeros(outputImageDim,outputImageDim);
	arma::mat temp_pool_id = arma::zeros(outputImageDim,outputImageDim); 
	sampleLoc = arma::zeros(outputImageDim*outputImageDim*inputImageNum,samples_num);
	for(int i = 0;i < samples_num;i++){
		for(int j = 0;j < inputImageNum;j++){
			arma::mat image = data.col(i).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
			image.reshape(inputImageDim,inputImageDim);
			for(int poolrow = 0; poolrow < outputImageDim; poolrow ++){
				int offsetrow = poolrow*poolingDim;
				for(int poolcol = 0;poolcol < outputImageDim; poolcol++){
					int offsetcol = poolcol*poolingDim;
					arma::mat patch = image.submat(offsetrow,offsetcol,offsetrow+poolingDim-1,offsetcol+poolingDim-1);
					if(poolingType == "MEAN"){
						temp_pooling_result(poolrow,poolcol) = arma::sum(arma::sum(patch))/patch.size();
						
					}else if(poolingType == "STOCHASTIC"){
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
			pooling_result.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = temp_pooling_result;
			sampleLoc.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = temp_pool_id;
		}


	}

	return pooling_result;
}

arma::mat Pooling::backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param){
	const int samples_num = delta.n_cols;
	
	arma::mat curr_delta = arma::zeros(inputImageDim*inputImageDim*outputImageNum,samples_num);

	arma::mat temp_delta = arma::zeros(inputImageDim,inputImageDim);
	arma::mat temp_last_delta = arma::zeros(inputImageDim,inputImageDim);
	for(int i = 0;i < samples_num;i++){
		for(int j = 0;j < outputImageNum;j++){
			temp_delta = curr_delta.col(i).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
			temp_last_delta = delta.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1);
			temp_delta.reshape(inputImageDim,inputImageDim);
			temp_last_delta.reshape(outputImageDim,outputImageDim);
			if(poolingType == "MEAN"){
				temp_delta = kron(temp_last_delta,ones(poolingDim));
				temp_delta /= (poolingDim*poolingDim);
			}else{
				for(int row = 0;row < temp_last_delta.n_rows; row ++){
					for(int col = 0;col < temp_last_delta.n_cols;col++){
						temp_delta(sampleLoc(row,col)) = temp_last_delta(row,col);
					}
				}


			}
		}
	}
	return curr_delta;

}
