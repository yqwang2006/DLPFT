#include "Pooling.h"
using namespace dlpft::module;
//#define __CUDA__
#ifdef __CUDA__
#include "../util/cupooling.h"
#endif
arma::mat Pooling::down_sample(arma::mat data){
	clock_t start_time = clock();
	clock_t end_time;
	double duration = 0;


	const int samples_num = data.n_cols;
	arma::mat patch,sto_patch ;

	arma::mat pooling_result = arma::zeros(outputImageDim*outputImageDim*inputImageNum,samples_num);
	arma::mat temp_pooling_result = arma::zeros(outputImageDim,outputImageDim);
	arma::mat temp_pool_id = arma::zeros(outputImageDim,outputImageDim); 
	arma::mat image = zeros(inputImageDim,inputImageDim);
	sampleLoc = arma::zeros(outputImageDim*outputImageDim*inputImageNum,samples_num);

	if(poolingType == "MEAN"){
		for(int i = 0;i < samples_num;i++){
			for(int j = 0;j < inputImageNum;j++){
				image = data.col(i).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
				image.reshape(inputImageDim,inputImageDim);


				for(int poolrow = 0; poolrow < outputImageDim; poolrow ++){
					int offsetrow = poolrow*poolingDim;
					for(int poolcol = 0;poolcol < outputImageDim; poolcol++){
						int offsetcol = poolcol*poolingDim;
						patch = image.submat(offsetrow,offsetcol,offsetrow+poolingDim-1,offsetcol+poolingDim-1);

						temp_pooling_result(poolrow,poolcol) = arma::accu(patch)/patch.size();
					}
				}

				pooling_result.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = reshape(temp_pooling_result,temp_pooling_result.size(),1);
				sampleLoc.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = reshape(temp_pool_id,temp_pool_id.size(),1);
			}
		}
	}else if(poolingType == "STOCHASTIC"){
		for(int i = 0;i < samples_num;i++){
			for(int j = 0;j < inputImageNum;j++){
				image = data.col(i).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
				image.reshape(inputImageDim,inputImageDim);


				for(int poolrow = 0; poolrow < outputImageDim; poolrow ++){
					int offsetrow = poolrow*poolingDim;
					for(int poolcol = 0;poolcol < outputImageDim; poolcol++){
						int offsetcol = poolcol*poolingDim;
						patch = image.submat(offsetrow,offsetcol,offsetrow+poolingDim-1,offsetcol+poolingDim-1);
						sto_patch = (double)1/(arma::accu(patch)) * patch ;
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

					}
				}

				pooling_result.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = reshape(temp_pooling_result,temp_pooling_result.size(),1);
				sampleLoc.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = reshape(temp_pool_id,temp_pool_id.size(),1);
			}


		}
	}else{
#ifdef __CUDA__
	//cumaxpooling(double* all_images, double* pooling_result, int* pooling_loc, int samples_num, int input_image_dim, int input_image_num, int output_image_dim, int pooling_dim);
		double *all_images = new double[samples_num * inputImageNum * inputImageDim * inputImageDim];
		double *h_pooling_result = new double[samples_num * inputImageNum * outputImageDim * outputImageDim];
		int *pooling_loc = new int[samples_num * inputImageNum * outputImageDim * outputImageDim];
		for(int i = 0;i < samples_num; i++){
			int start_i_loc = i*inputImageNum * inputImageDim * inputImageDim;
			for(int j = 0; j < inputImageNum; j++){
				int start_j_loc = j * inputImageDim * inputImageDim;
				for(int x = 0; x < inputImageDim; x++){
					for(int y = 0; y < inputImageDim; y++){
						all_images[start_i_loc + start_j_loc + x * inputImageDim + y] = data(start_j_loc+y*inputImageDim+x,i);
					}
				}
			}
			
		}
		cumaxpooling(all_images,h_pooling_result,pooling_loc,samples_num,inputImageDim,inputImageNum,outputImageDim,poolingDim);
		
		for(int i = 0;i < samples_num; i++){
			int start_i_loc = i*inputImageNum * outputImageDim * outputImageDim;
			for(int j = 0; j < outputImageNum; j++){
				int start_j_loc = j * outputImageDim * outputImageDim;
				for(int x = 0; x < outputImageDim; x++){
					for(int y = 0; y < outputImageDim; y++){
						pooling_result(start_j_loc+x*outputImageDim+y,i) = h_pooling_result[start_i_loc + start_j_loc + x * outputImageDim + y];
						sampleLoc(start_j_loc+x*outputImageDim+y,i) = pooling_loc[start_i_loc + start_j_loc + x * outputImageDim + y];
							
					}
				}
			}
			
		}


#else
		for(int i = 0;i < samples_num;i++){
			for(int j = 0;j < inputImageNum;j++){
				image = data.col(i).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
				image.reshape(inputImageDim,inputImageDim);


				for(int poolrow = 0; poolrow < outputImageDim; poolrow ++){
					int offsetrow = poolrow*poolingDim;
					for(int poolcol = 0;poolcol < outputImageDim; poolcol++){
						int offsetcol = poolcol*poolingDim;
						patch = image.submat(offsetrow,offsetcol,offsetrow+poolingDim-1,offsetcol+poolingDim-1);


						double max_patch_val = patch(0);// = arma::max(arma::max(patch));
						int max_patch_loc = 0;
						for(int i = 1;i < patch.size(); i++){
							if(max_patch_val < patch(i)){
								max_patch_val = patch(i);
								max_patch_loc = i;
							}
						}
						temp_pooling_result(poolrow,poolcol) = max_patch_val;
						//arma::uvec id = arma::find(patch==max_patch_val);
						//int max_in_patch_col = id(0)/patch.n_rows;
						//int max_in_patch_row = id(0)-patch.n_rows*max_in_patch_col;
						int max_in_patch_col = max_patch_loc/patch.n_rows;
						int max_in_patch_row = max_patch_loc-patch.n_rows*max_in_patch_col;
						int max_in_image_col = offsetcol + max_in_patch_col;
						int max_in_image_row = offsetrow + max_in_patch_row;
						int max_id_in_image = max_in_image_col * inputImageDim + max_in_image_row;
						temp_pool_id(poolrow,poolcol) = max_id_in_image;


					}
				}

				pooling_result.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = reshape(temp_pooling_result,temp_pooling_result.size(),1);
				sampleLoc.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1) = reshape(temp_pool_id,temp_pool_id.size(),1);
			}


		}
#endif
	}

	return pooling_result;
}
arma::mat Pooling::forwardpropagate(const arma::mat data,  NewParam param){
	arma::mat sample_data = down_sample(data);

	return sample_data;
}
arma::mat Pooling::process_delta(arma::mat curr_delta){
	const int samples_num = curr_delta.n_cols;
	arma::mat up_sampling_delta = arma::zeros(inputImageDim*inputImageDim*outputImageNum,samples_num);
	arma::mat temp_delta = arma::zeros(inputImageDim,inputImageDim);
	arma::mat temp_curr_delta = arma::zeros(inputImageDim,inputImageDim);
	arma::mat temp_pool_id;

	for(int i = 0;i < samples_num;i++){
		for(int j = 0;j < outputImageNum;j++){


			temp_curr_delta = curr_delta.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1);
			temp_delta = arma::zeros(inputImageDim,inputImageDim);
			temp_curr_delta.reshape(outputImageDim,outputImageDim);


			if(poolingType == "MEAN"){
				temp_delta = kron(temp_curr_delta,ones(poolingDim,poolingDim));
				temp_delta /= (poolingDim*poolingDim);
			}else{

				temp_pool_id = sampleLoc.col(i).rows(j*outputImageDim*outputImageDim,(j+1)*outputImageDim*outputImageDim-1);
				temp_pool_id.reshape(outputImageDim,outputImageDim);
				for(int row = 0;row < temp_curr_delta.n_rows; row ++){
					for(int col = 0;col < temp_curr_delta.n_cols;col++){
						temp_delta(temp_pool_id(row,col)) = temp_curr_delta(row,col);
					}
				}


			}


			temp_delta.reshape(inputImageDim*inputImageDim,1);
			up_sampling_delta.col(i).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1)= temp_delta;
		}
	}
	return up_sampling_delta;

}
arma::mat Pooling::backpropagate(const arma::mat next_delta, const arma::mat features, NewParam param){

	arma::mat curr_delta = next_delta; 

	//arma::mat curr_delta = active_function_dev(activeFuncChoice,features) % next_delta; 

	return curr_delta;
}
void Pooling::initial_weights_bias(){
	//weightMatrix = 0.005*arma::randu<arma::mat> (outputImageNum,1);
	//bias = zeros(outputImageNum,1);
	if(load_weight == "YES"){
		if(weight_addr != "" && bias_addr != ""){
			if(initial_weights_bias_from_file(weight_addr,bias_addr)){
				return;
			}
		}
	}

	weightMatrix = zeros(outputImageNum,1);
	bias = zeros(outputImageNum,1);

}
void Pooling::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,double weight_decay,arma::mat& Wgrad, arma::mat& bgrad){
	bgrad.set_size(outputImageNum,1);
	Wgrad.set_size(outputImageNum,1);
	//arma::mat down_sample_data = down_sample(input_data);
	Wgrad = zeros(outputImageNum,1);
	bgrad = zeros(outputImageNum,1);
	//#if DEBUG
	//	Wgrad = zeros(outputImageNum,1);
	//	bgrad = zeros(outputImageNum,1);
	//#else
	//	for(int i = 0;i < outputImageNum; i++){
	//		
	//		Wgrad(i) = sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)%down_sample_data.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
	//		bgrad(i) = sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
	//	}
	//#endif

}