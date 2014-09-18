#include "RBM.h"
#include <assert.h>
using namespace dlpft::module;
using namespace dlpft::param;
void RBM::pretrain(const arma::mat data,NewParam param){
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
			
			
			CD_k(1,minibatches[batch],c_bias);
			
			
			if(batch < 5)
				momentum = inittialmomentum;
			else
				momentum = finalmomentum;
			
			
			//update W,b,c
			
			deltaW = momentum * deltaW + learn_rate *
				((h_means * minibatches[batch].t() - nh_means * nv_means.t())/batch_size - weightcost * weightMatrix);
			deltac = momentum * deltac + (learn_rate/batch_size) * 
				(sum(minibatches[batch],1) - sum(nv_means,1));
			deltab = momentum * deltab + (learn_rate/batch_size) *
				(sum(h_means,1) - sum(nh_means,1));


			weightMatrix += deltaW ;
			bias += deltab ;
			c_bias = c_bias + deltac ;

			double err = arma::sum(arma::sum(arma::pow((minibatches[batch]-nv_means),2)));
			errsum += err;
		}

		LogOut << "Ended epoch " << epoch << "/" << max_epoch << ". Reconstruction error is " << errsum << endl;
		cout << "Ended epoch " << epoch << "/" << max_epoch << ". Reconstruction error is " << errsum << endl;

	}

	//result_model.features = RBM_VtoH(data, result_model);

	delete []minibatches;
}
arma::mat RBM::forwardpropagate(const arma::mat data,  NewParam param){
	//weightMat: hidden_size * visible_size
	//bias: (hidden_size,1)
	arma::mat activation = weightMatrix * data + repmat(bias,1,data.n_cols);
	activation = active_function(activeFuncChoice,activation);
	return activation;
}
arma::mat RBM::backpropagate(arma::mat next_layer_weight,const arma::mat next_delta, const arma::mat features, NewParam param){
	arma::mat curr_delta = active_function_dev(activeFuncChoice,features) % next_delta; 
	return curr_delta;
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
void RBM::sample_h_given_v(arma::mat& v0_sample, arma::mat& mean, arma::mat& sample){
	mean = propup(v0_sample);
	sample = BiNomial(mean);
}
void RBM::sample_v_given_h(arma::mat& h0_sample, arma::mat& mean, arma::mat& sample, arma::mat& v_bias){
	mean = propdown(h0_sample,v_bias);
	sample = BiNomial(mean);
}
arma::mat RBM::propup(arma::mat& v){
	assert(weightMatrix.n_cols == v.n_rows);
	arma::mat negh = weightMatrix * v + arma::repmat(bias,1,v.n_cols);
	negh = active_function(activeFuncChoice,negh);

	return negh;
}
arma::mat RBM::propdown(arma::mat& h,arma::mat& v_bias){
	assert(h.n_rows == weightMatrix.n_rows);
	arma::mat negdata = weightMatrix.t() * h + arma::repmat(v_bias,1,h.n_cols);
	negdata = active_function(activeFuncChoice,negdata);
	return negdata;
}
void  RBM::gibbs_hvh( arma::mat& v_bias,arma::mat& h0_sample){
	sample_v_given_h(h0_sample, nv_means, nv_samples, v_bias);
	sample_h_given_v(nv_samples, nh_means, nh_samples);
}
double RBM::get_reconstruct_error(arma::mat& v){
	return 0;
}
void  RBM::CD_k(int k,arma::mat& v, arma::mat& v_bias){
	sample_h_given_v(v, h_means, h_samples);
	for(int step = 0;step < k; step++){
		if(step == 0){
			gibbs_hvh(v_bias,h_samples);
		}else{
			gibbs_hvh(v_bias,nh_samples);
		}
	}
}
void RBM::initial_weights_bias(){
	if(load_weight == "YES"){
		if(weight_addr != "" && bias_addr != ""){
			if(initial_weights_bias_from_file(weight_addr,bias_addr)){
				return;
			}
		}
	}
		srand(unsigned(time(NULL)));
		weightMatrix = 0.1 * arma::randn(outputSize,inputSize);
		bias = arma::zeros(outputSize,1);
	
}
void RBM::calculate_grad_using_delta(const arma::mat input_data,const arma::mat delta,NewParam param,double weight_decay,arma::mat& Wgrad, arma::mat& bgrad){
	int lambda = atoi(param.params[params_name[WEIGHTDECAY]].c_str());
	Wgrad = ((double)1/input_data.n_cols)*delta * input_data.t();// + 0.003 * weightMatrix;
	bgrad = sum(delta,1)/input_data.n_cols;
}