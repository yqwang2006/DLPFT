#include "RBM.h"
#include <assert.h>
using namespace dlpft::module;
using namespace dlpft::param;
ResultModel RBM::pretrain(const arma::mat data, const arma::imat labels, NewParam param){
	ResultModel result_model;
	int hid_size = atoi(param.params[params_name[HIDNUM]].c_str());
	int max_epoch = atoi(param.params[params_name[MAXEPOCH]].c_str());
	int batch_size = atoi(param.params[params_name[BATCHSIZE]].c_str());
	double learn_rate = atof(param.params[params_name[LEARNRATE]].c_str());

	double inittialmomentum = 0.5;
	double finalmomentum = 0.9;
	double momentum = 0;

	double weightcost = 0.0002;

	int sample_num = data.n_cols;
	int visible_size = data.n_rows;
	int num_batches = sample_num / batch_size;
	
	result_model.weightMatrix = 0.1 * arma::randn(hid_size,visible_size);
	result_model.bias = arma::zeros(hid_size,1);
	arma::mat c_bias = arma::zeros(visible_size,1);

	nv_means = arma::zeros(visible_size,batch_size);
	nv_samples = arma::zeros(visible_size,batch_size);
	nh_means = arma::zeros(hid_size,batch_size);
	nh_samples = arma::zeros(hid_size,batch_size);
	h_means = arma::zeros(hid_size,batch_size);
	h_samples = arma::zeros(hid_size,batch_size);

	arma::mat* minibatches = new arma::mat[num_batches];

	rand_data(data,minibatches,sample_num,batch_size);

	arma::mat deltaW = zeros(hid_size,visible_size);
	arma::mat deltab = zeros(hid_size,1);
	arma::mat deltac = zeros(visible_size,1);

	double errsum = 0;

	for(int epoch = 0; epoch < max_epoch; epoch++){
		errsum = 0;
		for(int batch = 0; batch < num_batches; batch++){
			CD_k(1,minibatches[batch],result_model.weightMatrix,result_model.bias,c_bias);
			
			
			if(batch < 5)
				momentum = inittialmomentum;
			else
				momentum = finalmomentum;
			
			
			//update W,b,c
			
			deltaW = momentum * deltaW + learn_rate *
				((h_means * minibatches[batch].t() - nh_means * nv_samples.t())/batch_size - weightcost * result_model.weightMatrix);
			deltac = momentum * deltac + (learn_rate/batch_size) * 
				(sum(minibatches[batch],1) - sum(nv_samples,1));
			deltab = momentum * deltab + (learn_rate/batch_size) *
				(sum(h_means,1) - sum(nh_means,1));


			result_model.weightMatrix = result_model.weightMatrix + deltaW ;
			result_model.bias = result_model.bias + deltab ;
			c_bias = c_bias + deltac ;

			double err = arma::sum(arma::sum(arma::pow((minibatches[batch]-nv_means),2)));
			errsum += err;
		}

		cout << "Ended epoch " << epoch << "/" << max_epoch << ". Reconstruction error is " << errsum << endl;

	}

	//result_model.features = RBM_VtoH(data, result_model);

	//delete minibatches;

	return result_model;
}
arma::mat RBM::backpropagate(ResultModel& result_model,const arma::mat delta, const arma::mat feature, arma::imat labels, NewParam param){
	arma::mat errsum;
	arma::mat curr_delta;
	errsum = result_model.weightMatrix.t() * delta;

	curr_delta = active_function_inv(active_func_choice,feature) % errsum; 
	return curr_delta;
}
arma::mat RBM::forwardpropagate(const ResultModel result_model,const arma::mat data, const arma::imat labels, NewParam param){
	arma::mat features = result_model.weightMatrix * data + arma::repmat(result_model.bias,1,data.n_cols);
	features = active_function(active_func_choice,features);
	return features;
}


arma::mat RBM::BiNomial(const arma::mat mean){
	arma::mat rand_vec = arma::randu(mean.n_rows,mean.n_cols);
	arma::uvec indeies = find(mean>rand_vec);
	arma::mat result = arma::zeros(mean.n_rows,mean.n_cols);
	for(int i = 0;i < indeies.size();i++){
		//error
		result(indeies(i)) = 1;
	}
	return result;
}
void RBM::sample_h_given_v(arma::mat& v0_sample, arma::mat& mean, arma::mat& sample,arma::mat& weightMat, arma::mat& h_bias){
	mean = propup(v0_sample,weightMat,h_bias);
	sample = BiNomial(mean);
}
void RBM::sample_v_given_h(arma::mat& h0_sample, arma::mat& mean, arma::mat& sample,arma::mat& weightMat, arma::mat& v_bias){
	mean = propdown(h0_sample,weightMat,v_bias);
	sample = BiNomial(mean);
}
arma::mat RBM::propup(arma::mat& v,arma::mat& weightMat, arma::mat& h_bias){
	assert(weightMat.n_cols == v.n_rows);
	arma::mat negdata = weightMat * v + arma::repmat(h_bias,1,v.n_cols);
	negdata = active_function(active_func_choice,negdata);

	return negdata;
}
arma::mat RBM::propdown(arma::mat& h,arma::mat& weightMat,arma::mat& v_bias){
	assert(h.n_rows == weightMat.n_rows);
	arma::mat negh = weightMat.t() * h + arma::repmat(v_bias,1,h.n_cols);
	negh = active_function(active_func_choice,negh);
	return negh;
}
void  RBM::gibbs_hvh(arma::mat& weightMat, arma::mat& h_bias, arma::mat& v_bias,arma::mat& h0_sample){
	sample_v_given_h(h0_sample, nv_means, nv_samples,weightMat, v_bias);
	sample_h_given_v(nv_samples, nh_means, nh_samples,weightMat,h_bias);
}
double RBM::get_reconstruct_error(arma::mat& v){
	return 0;
}
void  RBM::CD_k(int k,arma::mat& v, arma::mat& weightMat, arma::mat& h_bias, arma::mat& v_bias){
	sample_h_given_v(v, h_means, h_samples,weightMat,h_bias);
	for(int step = 0;step < k; step++){
		if(step == 0){
			gibbs_hvh(weightMat,h_bias,v_bias,h_samples);
		}else{
			gibbs_hvh(weightMat,h_bias,v_bias,nh_samples);
		}
	}
}