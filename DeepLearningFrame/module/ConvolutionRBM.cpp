#include "ConvolutionRBM.h"
#include "../util/randdata.h"
using namespace dlpft::module;
void ConvolutionRBM::initial_weights_bias(){
	bias = zeros(outputImageNum,1);
	weightMatrix = zeros(filterDim*filterNum,filterDim);
#if DEBUG
	for(int i = 0; i < filterNum; i++){
		weightMatrix.rows(i*filterDim,(i+1)*filterDim-1) = (i+1)*0.1*ones(filterDim,filterDim);
	}
#else
	cube tempW = 0.1 * randn(filterDim,filterDim,filterNum);
	for(int i = 0; i < filterNum; i++){
		weightMatrix.rows(i*filterDim,(i+1)*filterDim-1) = tempW.slice(i);
	}
#endif
	
	
}
void ConvolutionRBM::pretrain(const arma::mat data, const arma::imat labels, NewParam param){
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
			CD_k(1,minibatches[batch],v_bias);
		}
	}



}
void  ConvolutionRBM::CD_k(int k,arma::mat& v, double v_bias){
	arma::mat h_means, h_samples,nh_samples;
	sample_h_given_v(v, h_means, h_samples);
	for(int step = 0;step < k; step++){
		if(step == 0){
			gibbs_hvh(v_bias,h_samples);
		}else{
			gibbs_hvh(v_bias,nh_samples);
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
arma::mat ConvolutionRBM::propup(arma::mat& v){
	//assert(weightMat.n_cols == v.n_rows);
	arma::mat negdata = weightMatrix * v + arma::repmat(bias,1,v.n_cols);
	negdata = active_function(activeFuncChoice,negdata);

	return negdata;
}
arma::mat ConvolutionRBM::propdown(arma::mat& h,double v_bias){
	//assert(h.n_rows == weightMat.n_rows);
	arma::mat negh = weightMatrix.t() * h + v_bias;
	//negh = active_function(activeFuncChoice,negh);
	return negh;
}
void  ConvolutionRBM::gibbs_hvh(double v_bias,arma::mat& h0_sample){
	//sample_v_given_h(h0_sample, nv_means, nv_samples,weightMat, v_bias);
	//sample_h_given_v(nv_samples, nh_means, nh_samples,weightMat,h_bias);
}
arma::mat ConvolutionRBM::forwardpropagate(const arma::mat data,  NewParam param){
	const int samples_num = data.n_cols;

	arma::mat all_features = arma::zeros(outputImageDim*outputImageDim*outputImageNum,samples_num);
	mat features_filter = zeros(outputImageDim,outputImageDim);
	mat W = zeros(filterDim,filterDim);
	mat image = zeros(inputImageDim*inputImageDim,1);
	for(int n = 0;n < samples_num; n++){
		
		
		for(int nout = 0; nout < outputImageNum; nout ++){
			features_filter = zeros(outputImageDim,outputImageDim);
			int fmInBase = 0;
			double b = (bias.row(nout))(0);
			for(int nin = 0; nin < inputImageNum; nin++){
				W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				image = data.col(n).rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1);
				image.reshape(inputImageDim,inputImageDim);
				features_filter += convn(image,W,"valid");
			}
			features_filter = features_filter + b;
			features_filter = active_function(activeFuncChoice,features_filter);
			all_features.col(n).rows(nout*features_filter.size(),(nout+1)*features_filter.size()-1) = reshape(features_filter,features_filter.size(),1);
		}
		
	}

	return all_features;
}
arma::mat ConvolutionRBM::process_delta(arma::mat curr_delta){
	const int samples_num = curr_delta.n_cols;
	int delta_dim = sqrt(curr_delta.n_rows / outputImageNum);
	arma::mat convn_delta = zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	//int conv_dim = delta_dim - filter_dim + 1;

	//arma::mat curr_delta = arma::zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	
	for(int n = 0;n < samples_num; n++){
		
		for(int nin = 0; nin < inputImageNum; nin++){
			int fmInBase = 0;
			arma::mat delta_filter = zeros(inputImageDim,inputImageDim);
			for(int nout = 0; nout < outputImageNum; nout ++){
				double b = (bias.row(nout))(0);
				arma::mat W = weightMatrix.rows(filterDim * (nout*inputImageNum + nin),filterDim * (nout*inputImageNum + nin + 1)-1);
				
				arma::mat single_delta = curr_delta.col(n).rows(nout*delta_dim*delta_dim,(nout+1)*delta_dim*delta_dim-1);
				single_delta.reshape(delta_dim,delta_dim);
				delta_filter += convn(single_delta,W,"full");
			}
			convn_delta.col(n).rows(nin*delta_filter.size(),(nin+1)*delta_filter.size()-1) = reshape(delta_filter,delta_filter.size(),1);
		}
		
	}
	return convn_delta;
}
arma::mat ConvolutionRBM::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	//���������һ��һ����pooling�㣬pooling��͵�ǰ�ľ��������maps����һ��
	//ͬʱpooling���ÿ��map����һ������beta(����������е�weightMatrix(i)����bias(i)
	//֮����������maps
	
	arma::mat curr_delta = zeros(outputSize,next_delta.n_cols);
	for(int i = 0;i < outputImageNum;i ++){
//#if DEBUG
		curr_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1) 
			= (active_function_dev(activeFuncChoice,features.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)) 
			% next_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1));
//#else 
//		curr_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1) 
//			= next_layer_weight(i)
//			*(active_function_dev(activeFuncChoice,features.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)) 
//			% next_delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1));
//#endif
	}
	
	return curr_delta;
	

}
void ConvolutionRBM::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta, NewParam param,arma::mat& Wgrad, arma::mat& bgrad){
	//compute bgrad
	int mbSize = input_data.n_cols;
	double lambda = 3e-3;
	Wgrad.set_size(filterDim*filterNum,filterDim);
	bgrad.set_size(outputImageNum,1);
	
	for(int i = 0; i < outputImageNum; i++){
		for(int j = 0;j < inputImageNum;j++){
			mat Wgrad_j_i = zeros(filterDim,filterDim);
			for(int k = 0;k < mbSize; k++){
				mat input_image = input_data.col(k).rows(j*inputImageDim*inputImageDim,(j+1)*inputImageDim*inputImageDim-1);
				mat delta_i_k = delta.col(k).rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1);

				input_image.reshape(inputImageDim,inputImageDim);
				delta_i_k.reshape(outputImageDim,outputImageDim);
				Wgrad_j_i += convn(input_image,delta_i_k,"valid");
			}
			Wgrad.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum+j+1)-1) = ((double)1/mbSize)*Wgrad_j_i
				+lambda*weightMatrix.rows(filterDim * (i*inputImageNum + j),filterDim * (i*inputImageNum + j + 1)-1);
		}

		bgrad(i) = ((double)1/mbSize)*sum(sum(delta.rows(i*outputImageDim*outputImageDim,(i+1)*outputImageDim*outputImageDim-1)));
	}
}