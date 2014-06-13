#include "Pooling.h"
using namespace dlpft::module;
arma::mat Pooling::down_sample(arma::mat data){
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
			
			pooling_result.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = reshape(temp_pooling_result,temp_pooling_result.size(),1);
			sampleLoc.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = reshape(temp_pool_id,temp_pool_id.size(),1);
		}


	}

	return pooling_result;
}
arma::mat Pooling::forwardpropagate(const arma::mat data,  NewParam param){
	arma::mat sample_data = down_sample(data);
	for(int j = 0;j < outputImageNum; j++){
		double W = weightMatrix(j);
		double b = bias(j);
		sample_data.rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = W*sample_data.rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) + b;
		sample_data.rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = active_function(activeFuncChoice,sample_data.rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1));
	}
	return sample_data;
}
arma::mat Pooling::process_delta(arma::mat curr_delta){
	const int samples_num = curr_delta.n_cols;
	arma::mat up_sampling_delta = arma::zeros(inputImageDim*inputImageDim*outputImageNum,samples_num);
	arma::mat temp_delta = arma::zeros(inputImageDim,inputImageDim);
	arma::mat temp_curr_delta = arma::zeros(inputImageDim,inputImageDim);
	for(int i = 0;i < samples_num;i++){
		for(int j = 0;j < outputImageNum;j++){
			
			temp_curr_delta = curr_delta.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1);
			temp_delta.reshape(inputImageDim,inputImageDim);
			temp_curr_delta.reshape(outputImageDim,outputImageDim);
			if(poolingType == "MEAN"){
				temp_delta = kron(temp_curr_delta,ones(poolingDim));
				temp_delta /= (poolingDim*poolingDim);
			}else{
				for(int row = 0;row < temp_curr_delta.n_rows; row ++){
					for(int col = 0;col < temp_curr_delta.n_cols;col++){
						temp_delta(sampleLoc(row,col)) = temp_curr_delta(row,col);
					}
				}


			}
			temp_delta.reshape(inputImageDim*inputImageDim,1);
			up_sampling_delta.col(i).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1)= temp_delta;
		}
	}
	return up_sampling_delta;
	
}
arma::mat Pooling::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	arma::mat curr_delta = active_function_dev(activeFuncChoice,features) % next_delta; 
	return curr_delta;
}
void Pooling::initial_weights_bias(){
	weightMatrix = 0.005*arma::randu<arma::mat> (outputImageNum,1);
	bias = zeros(outputImageNum,1);
}
void Pooling::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,arma::mat& Wgrad, arma::mat& bgrad){
	bgrad.set_size(outputImageNum,1);
	Wgrad.set_size(outputImageNum,1);
	arma::mat down_sample_data = down_sample(input_data);

	for(int i = 0;i < outputImageNum; i++){
		
		Wgrad(i) = sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)%down_sample_data.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
		bgrad(i) = sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
	}

}