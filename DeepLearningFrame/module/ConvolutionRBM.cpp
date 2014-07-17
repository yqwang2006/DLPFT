#include "ConvolutionRBM.h"
#include "../util/randdata.h"
using namespace dlpft::module;
void ConvolutionRBM::initial_weights_bias(){
	bias = zeros(outputImageNum,1);
	weightMatrix = zeros(filterDim*filterNum,filterDim);
#if DEBUG
	for(int i = 0; i < filterNum; i++){
		weightMatrix.rows(i*filterDim,(i+1)*filterDim-1) = (i+1)*ones(filterDim,filterDim);
	}
#else
	cube tempW = 0.1 * randn(filterDim,filterDim,filterNum);
	for(int i = 0; i < filterNum; i++){
		weightMatrix.rows(i*filterDim,(i+1)*filterDim-1) = tempW.slice(i);
	}
#endif
	
	
}
void ConvolutionRBM::pretrain(const arma::mat data, NewParam param){
	int max_epoch = atoi(param.params[params_name[MAXEPOCH]].c_str());
	int batch_size = atoi(param.params[params_name[BATCHSIZE]].c_str());
	double learn_rate = atof(param.params[params_name[LEARNRATE]].c_str());
	
	int sample_num = data.n_cols;
	int num_batches = sample_num / batch_size;

	double v_bias = 0;

	arma::mat* minibatches = new arma::mat[num_batches];

	rand_data(data,minibatches,sample_num,batch_size);

	for(int epoch = 1; epoch <=  max_epoch; epoch++){
		double errsum = 0;
		for(int batch = 0; batch < num_batches; batch++){



			double error = 0;
			arma::mat Wgrad = zeros(weightMatrix.n_rows,weightMatrix.n_cols);
			arma::mat hgrad = zeros(outputImageNum,1);
			double vgrad = 0;
			crbmGradients(1,minibatches[batch],param,v_bias,Wgrad,hgrad,vgrad,error);


			weightMatrix += learn_rate * Wgrad;

			
			bias += learn_rate * hgrad;
			v_bias += learn_rate * vgrad;

			LogOut << "Ended batch " << batch+1 << "/" << num_batches << ". Reconstruction error is " << error << endl;
			cout << "Ended batch " << batch+1 << "/" << num_batches << ". Reconstruction error is " << error << endl;
			errsum += error;
		}

		LogOut << "Ended epoch " << epoch << "/" << max_epoch << ". Reconstruction error is " << errsum << endl;
		cout << "Ended epoch " << epoch << "/" << max_epoch << ". Reconstruction error is " << errsum << endl;

	}

	delete[] minibatches;

}
void ConvolutionRBM::crbmGradients(int k,arma::mat minibatch,NewParam param,double v_bias, arma::mat& Wgrad, arma::mat& hgrad, double& vgrad, double& error){
	arma::mat h_means, h_samples,nh_samples,nv_means,nv_samples,nh_means;
	
	CD_k(1,minibatch,v_bias, h_means, h_samples,nv_means,nv_samples,nh_means,nh_samples);

	arma::mat error_mat = arma::mean(pow(minibatch - nv_means,2),0);
	error = error_mat(0);
	
	arma::mat W2grad = zeros(weightMatrix.n_rows,weightMatrix.n_cols);
	arma::mat h2grad = zeros(outputImageNum,1);
	calculate_grad_using_delta(minibatch,h_means,param,Wgrad,hgrad);
	calculate_grad_using_delta(nv_means,nh_means,param,W2grad,h2grad);


	Wgrad = Wgrad - W2grad;
	hgrad = hgrad - h2grad;


	vgrad = ((double)1/minibatch.size())*sum(sum(minibatch-nv_means));



}

void  ConvolutionRBM::CD_k(int k,arma::mat& v, double v_bias, mat& h0_mean, mat& h0_samples, mat& nv_means,mat& nv_samples,mat& nh_means,mat& nh_samples){
	
	sample_h_given_v(v, h0_mean, h0_samples);
	for(int step = 0;step < k; step++){
		if(step == 0){
			gibbs_hvh(v_bias,h0_samples,nv_means,nv_samples,nh_means,nh_samples);
		}else{
			gibbs_hvh(v_bias,nh_samples,nv_means,nv_samples,nh_means,nh_samples);
		}
	}
}
arma::mat ConvolutionRBM::BiNomial(const arma::mat mean){
	arma::mat rand_vec = arma::randu(mean.n_rows,mean.n_cols);
	arma::uvec indeies = find(mean>rand_vec);
	arma::mat result = arma::zeros(mean.n_rows,mean.n_cols);
	for(int i = 0;i < indeies.size();i++){
		//error
		result(indeies(i)) = 1;
	}
	return result;
}
void ConvolutionRBM::sample_h_given_v(arma::mat& v0_sample, arma::mat& mean, arma::mat& sample){
	mean = propup(v0_sample);
	sample = BiNomial(mean);
}
void ConvolutionRBM::sample_v_given_h(arma::mat& h0_sample, arma::mat& mean, arma::mat& sample, double v_bias){
	mean = propdown(h0_sample,v_bias);
	
	sample = BiNomial(mean);
	
}
arma::mat ConvolutionRBM::propup(const arma::mat v){
	const int samples_num = v.n_cols;
	int outputMapSize = outputImageDim * outputImageDim;

	arma::mat all_features = arma::zeros(outputMapSize*outputImageNum,samples_num);
	cube features_filter = zeros(outputImageDim,outputImageDim,samples_num);
	mat W = zeros(filterDim,filterDim);
	mat images = zeros(inputImageDim*inputImageDim,samples_num);
	cube all_images = zeros(inputImageDim*inputImageDim,samples_num,1);


	for(int nout = 0; nout < outputImageNum; nout ++){
			features_filter = zeros(outputImageDim,outputImageDim,samples_num);
			int fmInBase = 0;
			double b = (bias.row(nout))(0);
			for(int nin = 0; nin < inputImageNum; nin++){
				W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				images = v.rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1);
				all_images = zeros(inputImageDim*inputImageDim,samples_num,1);
				all_images.slice(0) = images;
				all_images.reshape(inputImageDim,inputImageDim,samples_num);
				//reshape(images,inputImageDim,inputImageDim,inputImageNum);
				
				features_filter += convn_cube(all_images,W,"valid");
			}
			features_filter = features_filter + b;
			//features_filter = active_function(activeFuncChoice,features_filter);
			features_filter.reshape(outputMapSize,samples_num,1);
			arma::mat& temp_filter = features_filter.slice(0);
			all_features.rows(nout*outputMapSize,(nout+1)*outputMapSize-1) = temp_filter;
		}

	all_features = active_function(activeFuncChoice,all_features);
	return all_features;
	
}
arma::mat ConvolutionRBM::propdown(arma::mat& h,double v_bias){
	const int samples_num = h.n_cols;
	int delta_dim = sqrt(h.n_rows / outputImageNum);
	arma::mat v_expected = zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	//int conv_dim = delta_dim - filter_dim + 1;

	//arma::mat curr_delta = arma::zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	mat W = zeros(filterDim,filterDim);
	mat single_delta = zeros(delta_dim*delta_dim,samples_num);
	arma::cube all_deltas = zeros(delta_dim*delta_dim,samples_num,1);
	arma::cube delta_filter = zeros(inputImageDim,inputImageDim,samples_num);
		
		for(int nin = 0; nin < inputImageNum; nin++){
			int fmInBase = 0;
			delta_filter = zeros(inputImageDim,inputImageDim,samples_num);
			for(int nout = 0; nout < outputImageNum; nout ++){
				double b = (bias.row(nout))(0);
				W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				
				single_delta = h.rows(nout*delta_dim*delta_dim,(nout+1)*delta_dim*delta_dim-1);

				all_deltas = zeros(delta_dim*delta_dim,samples_num,1);
				all_deltas.slice(0) = single_delta;
				all_deltas.reshape(delta_dim,delta_dim,samples_num);
				delta_filter += convn_cube(all_deltas,W,"full");
			}

			delta_filter.reshape(inputImageDim*inputImageDim,samples_num,1);
			arma::mat &temp_delta = delta_filter.slice(0);
			v_expected.rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1) =temp_delta;
		}
		v_expected = v_expected + v_bias;
		v_expected = active_function(activeFuncChoice,v_expected);
	return v_expected;
}
void  ConvolutionRBM::gibbs_hvh(double v_bias,arma::mat& h0_sample, arma::mat& nv_means, arma::mat& nv_samples, arma::mat& nh_means, arma::mat& nh_samples){
	sample_v_given_h(h0_sample, nv_means, nv_samples,v_bias);
	sample_h_given_v(nv_means, nh_means, nh_samples);
}
arma::mat ConvolutionRBM::forwardpropagate(const arma::mat data,  NewParam param){
	return propup(data);
}
arma::mat ConvolutionRBM::process_delta(arma::mat curr_delta){
	const int samples_num = curr_delta.n_cols;
	int delta_dim = sqrt(curr_delta.n_rows / outputImageNum);
	arma::mat convn_delta = zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	//int conv_dim = delta_dim - filter_dim + 1;

	//arma::mat curr_delta = arma::zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	
		
		for(int nin = 0; nin < inputImageNum; nin++){
			int fmInBase = 0;
			arma::cube delta_filter = zeros(inputImageDim,inputImageDim,samples_num);
			for(int nout = 0; nout < outputImageNum; nout ++){
				double b = (bias.row(nout))(0);
				arma::mat W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				
				arma::mat single_delta = curr_delta.rows(nout*delta_dim*delta_dim,(nout+1)*delta_dim*delta_dim-1);
				arma::cube all_deltas = zeros(delta_dim*delta_dim,samples_num,1);
				all_deltas.slice(0) = single_delta;
				all_deltas.reshape(delta_dim,delta_dim,samples_num);
				delta_filter += convn_cube(all_deltas,W,"full");
			}
			delta_filter.reshape(inputImageDim*inputImageDim,samples_num,1);
			arma::mat temp_delta = delta_filter.slice(0);
			convn_delta.rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1) =temp_delta;
		}
		
	
	return convn_delta;
}
arma::mat ConvolutionRBM::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	//卷基层的下一层一般是pooling层，pooling层和当前的卷基层输出maps个数一样
	//同时pooling层对每个map乘以一个常数beta(即下面代码中的weightMatrix(i)，和bias(i)
	//之后再输出多个maps
	
	arma::mat curr_delta = zeros(outputSize,next_delta.n_cols);
	curr_delta = active_function_dev(activeFuncChoice,features) % next_delta;
	
	return curr_delta;
	

}
void ConvolutionRBM::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta, NewParam param,arma::mat& Wgrad, arma::mat& bgrad){
//compute bgrad
	//compute bgrad
	clock_t start_time = clock();
	clock_t end_time;
	double duration = 0;
	
	int mbSize = input_data.n_cols;
	double lambda = 3e-3;
	Wgrad.set_size(filterDim*filterNum,filterDim);
	bgrad.set_size(outputImageNum,1);
	mat Wgrad_j_i = zeros(filterDim,filterDim);
	mat input_images = zeros(inputImageDim*inputImageDim,mbSize);
	mat delta_i_k = zeros(outputImageDim*outputImageDim,mbSize);
	cube all_images = arma::zeros(inputImageDim*inputImageDim,mbSize,1);
	cube all_delta = zeros(outputImageDim*outputImageDim,mbSize,1);
	for(int i = 0; i < outputImageNum; i++){
		for(int j = 0;j < inputImageNum;j++){
			Wgrad_j_i = zeros(filterDim,filterDim);
		
			input_images = input_data.rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
			delta_i_k = delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1);

			all_images = zeros(inputImageDim*inputImageDim,mbSize,1);
			all_images.slice(0) = input_images;
			all_images.reshape(inputImageDim,inputImageDim,mbSize);

			all_delta = zeros(outputImageDim*outputImageDim,mbSize,1);
			all_delta.slice(0) = delta_i_k;
			all_delta.reshape(outputImageDim,outputImageDim,mbSize);

			cube temp_wgrad = convn_cube(all_images,all_delta,"valid");

			for(int k = 0;k < temp_wgrad.n_slices; k++){
				Wgrad_j_i += temp_wgrad.slice(k);
			}
			
			
			Wgrad.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum+j+1)-1) 
				= ((double)1/mbSize)*Wgrad_j_i
				+lambda*weightMatrix.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum + j + 1)-1);
		}

		bgrad(i) = ((double)1/mbSize)*sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
	}
}