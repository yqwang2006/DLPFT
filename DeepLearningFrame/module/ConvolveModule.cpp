#include "ConvolveModule.h"
#include "../util/convolve.h"
using namespace dlpft::module;

arma::mat ConvolveModule::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	const int samples_num = data.n_cols;
	int filter_dim = atoi(param.params[params_name[FILTERDIM]].c_str());
	int filter_num = atoi(param.params[params_name[FILTERNUM]].c_str());
	int image_dim = sqrt(data.n_rows / lastFilterNum);
	int conv_dim = image_dim - filter_dim + 1;

	arma::mat all_features = arma::zeros(conv_dim*conv_dim*filter_num,samples_num);
	
	for(int n = 0;n < samples_num; n++){
		arma::mat features_filter = zeros(conv_dim,conv_dim);
		for(int nout = 0; nout < filter_num; nout ++){
			int fmInBase = 0;
			double bias = (result_model.bias.row(nout))(0);
			for(int nin = 0; nin < lastFilterNum; nin++){
				arma::mat W = result_model.weightMatrix.rows(filter_dim * (nout*filter_num + nin),filter_dim * (nout*filter_num + nin + 1)-1);
				arma::mat image = data.col(n).rows(nin*image_dim*image_dim,(nin+1)*image_dim*image_dim-1);
				image.reshape(image_dim,image_dim);
				features_filter += convn(image,W,"valid");
			}
			features_filter = features_filter + bias;
			features_filter.reshape(features_filter.size(),1);
			all_features.col(n).rows(nout*features_filter.size(),(nout+1)*features_filter.size()-1) = features_filter;
		}
		
	}

	return all_features;
}

arma::mat ConvolveModule::backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat features, const arma::imat labels,NewParam param){
	const int samples_num = delta.n_cols;
	int filter_dim = atoi(param.params[params_name[FILTERDIM]].c_str());
	int filter_num = atoi(param.params[params_name[FILTERNUM]].c_str());
	int delta_dim = sqrt(delta.n_rows / filter_num);
	//int conv_dim = delta_dim - filter_dim + 1;

	arma::mat curr_delta = arma::zeros(lastOutputDim*lastOutputDim*lastFilterNum,samples_num);
	
	for(int n = 0;n < samples_num; n++){
		arma::mat delta_filter = zeros(lastOutputDim,lastOutputDim);
		for(int nin = 0; nin < lastFilterNum; nin++){
			int fmInBase = 0;
			
			for(int nout = 0; nout < filter_num; nout ++){
				double bias = (result_model.bias.row(nout))(0);
				arma::mat W = result_model.weightMatrix.rows(filter_dim * (nout*filter_num + nin),filter_dim * (nout*filter_num + nin + 1)-1);
				


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
