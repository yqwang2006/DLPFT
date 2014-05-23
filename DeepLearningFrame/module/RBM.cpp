#include "RBM.h"
#include "../util/sigmoid.h"
#include <assert.h>
using namespace dlpft::module;
using namespace dlpft::param;
ResultModel RBM::run(arma::mat& data, arma::mat& labels, NewParam& param){
	ResultModel result_model;
	int hid_size = atoi(param.params["Hid_num"].c_str());
	int max_epoch = atoi(param.params["Max_epoch"].c_str());
	int batch_size = atoi(param.params["Batch_size"].c_str());
	double learn_rate = atof(param.params["Learning_rate"].c_str());
	
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



	double errsum = 0;

	for(int epoch = 0; epoch < max_epoch; epoch++){
		errsum = 0;
		for(int batch = 0; batch < num_batches; batch++){
			CD_k(1,minibatches[batch],result_model.weightMatrix,result_model.bias,c_bias);
			
			//update W,b,c
			
			arma::mat deltaW = h_means * minibatches[batch].t() - nh_means * nv_samples.t();
			arma::mat deltac = sum(minibatches[batch],1) - sum(nv_samples,1);
			arma::mat deltab = sum(h_means,1) - sum(nh_means,1);

			result_model.weightMatrix = result_model.weightMatrix + learn_rate *(deltaW / batch_size);
			result_model.bias = result_model.bias + learn_rate * (deltab / batch_size);
			c_bias = c_bias + learn_rate * (deltac / batch_size);

			double err = arma::sum(arma::sum(arma::pow((minibatches[batch]-nv_means),2)))/batch_size;
			errsum += err;
		}

		cout << "Ended epoch " << epoch << "/" << max_epoch << ". Reconstruction error is " << errsum << endl;

	}

	result_model.features = RBM_VtoH(data, result_model);

	//delete minibatches;

	return result_model;
}
void RBM::rand_data(arma::mat input, arma::mat* batches,int sample_num, int batch_size){
	
	srand(unsigned(time(NULL)));
	int batches_num = sample_num / batch_size;
	int visible_size = input.n_rows;
	vector<int> groups;
	for(int i = 0;i < batch_size; i++){
		for(int j = 0;j < batches_num; j++){
			groups.push_back(j);
		}
	}

	random_shuffle(groups.begin(),groups.end());

	arma::mat groups_mat = arma::zeros(groups.size(),1);
	for(int i = 0;i < groups.size(); i++)
	{
		groups_mat(i) = groups[i];
	}

	for(int i = 0;i < batches_num; i++){
		batches[i] = input.cols(find(groups_mat == i));
	}


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
	negdata = sigmoid(negdata);

	return negdata;
}
arma::mat RBM::propdown(arma::mat& h,arma::mat& weightMat,arma::mat& v_bias){
	assert(h.n_rows == weightMat.n_rows);
	arma::mat negh = weightMat.t() * h + arma::repmat(v_bias,1,h.n_cols);
	negh = sigmoid(negh);
	return negh;
}
void  RBM::gibbs_hvh(arma::mat& weightMat, arma::mat& h_bias, arma::mat& v_bias,arma::mat& h0_sample){
	sample_v_given_h(h0_sample, nv_means, nv_samples,weightMat, v_bias);
	sample_h_given_v(nv_samples, nh_means, nh_samples,weightMat,h_bias);
}
double RBM::get_reconstruct_error(arma::mat& v){
	return 0;
}
arma::mat RBM::RBM_VtoH(arma::mat& input,ResultModel& result_model){
	arma::mat result = result_model.weightMatrix * input + arma::repmat(result_model.bias,1,input.n_cols);
	result = sigmoid(result);
	return result;
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