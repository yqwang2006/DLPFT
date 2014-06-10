#include "ConvolveModule.h"
#include "../util/convolve.h"
using namespace dlpft::module;
void ConvolveModule::initial_params(){
	bias = zeros(outputImageDim,1);
	filters = 0.1 * randu<arma::mat> (filterDim*filterNum,filterDim);
	
}
arma::mat ConvolveModule::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	const int samples_num = data.n_cols;

	arma::mat all_features = arma::zeros(outputImageDim*outputImageDim*filterDim,samples_num);
	
	for(int n = 0;n < samples_num; n++){
		arma::mat features_filter = zeros(outputImageDim,outputImageDim);
		for(int nout = 0; nout < outputImageNum; nout ++){
			int fmInBase = 0;
			double b = (bias.row(nout))(0);
			for(int nin = 0; nin < inputImageNum; nin++){
				arma::mat W = filters.rows(filterDim * (nout*outputImageNum + nin),filterDim * (nout*outputImageNum + nin + 1)-1);
				arma::mat image = data.col(n).rows(nin*inputImageDim*inputImageDim,(nin+1)*inputImageDim*inputImageDim-1);
				image.reshape(inputImageDim,inputImageDim);
				features_filter += convn(image,W,"valid");
			}
			features_filter = features_filter + b;
			features_filter.reshape(features_filter.size(),1);
			all_features.col(n).rows(nout*features_filter.size(),(nout+1)*features_filter.size()-1) = features_filter;
		}
		
	}

	return all_features;
}

arma::mat ConvolveModule::backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param){
	const int samples_num = delta.n_cols;
	int delta_dim = sqrt(delta.n_rows / outputImageNum);
	//int conv_dim = delta_dim - filter_dim + 1;

	arma::mat curr_delta = arma::zeros(inputImageDim*inputImageDim*inputImageNum,samples_num);
	
	for(int n = 0;n < samples_num; n++){
		arma::mat delta_filter = zeros(inputImageDim,inputImageDim);
		for(int nin = 0; nin < inputImageNum; nin++){
			int fmInBase = 0;
			
			for(int nout = 0; nout < outputImageNum; nout ++){
				double b = (bias.row(nout))(0);
				arma::mat W = filters.rows(filterDim * (nout*outputImageNum + nin),filterDim * (nout*outputImageNum + nin + 1)-1);
				


				arma::mat single_delta = delta.col(n).rows(nout*delta_dim*delta_dim,(nout+1)*delta_dim*delta_dim-1);
				single_delta.reshape(delta_dim,delta_dim);
				delta_filter += convn(single_delta,W,"full");
			}
			delta_filter.reshape(delta_filter.size(),1);
			curr_delta.col(n).rows(nin*delta_filter.size(),(nin+1)*delta_filter.size()-1) = delta_filter;
		}
		
	}

	return curr_delta;

}
