#include "ModelCost.h"
#include "../util/onehot.h"
using namespace dlpft::module;
using namespace dlpft::param;
using namespace dlpft::function;
void ModelCost::initialParam(){
}
void ModelCost::modelff(const arma::mat inputdata,arma::mat *output,arma::mat* dropoutMask){
	double dropoutfraction = atof(params[layer_num].params[params_name[DROPOUTFRACTION]].c_str());
	mat print_data = inputdata;
	ofstream ofs;
	ofs.open("data.txt");
	print_data.quiet_save(ofs,raw_ascii);
	ofs.close();
	for(int i = 0;i < layer_num;i ++){
		stringstream str_i;
		str_i << i;
		std::string wname = "Weight_"+str_i.str() +".txt";
		string bname = "bias_"+str_i.str() + ".txt";
		string outname = "out_"+str_i.str()+".txt";
		ofs.open(wname);
		modules[i]->weightMatrix.quiet_save(ofs,raw_ascii);
		ofs.close();
		ofs.open(bname);
		modules[i]->bias.quiet_save(ofs,raw_ascii);
		ofs.close();
		if(i == 0){
			output[i] = modules[i]->forwardpropagate(data,params[i]);
		}else{
			output[i] = modules[i]->forwardpropagate(output[i-1],params[i]);
		}
		ofs.open(outname);
		output[i].quiet_save(ofs,raw_ascii);
		ofs.close();
		if(dropoutfraction > 0 && i != layer_num-1){
			arma::mat rand_mat = arma::randu(output[i].n_rows,output[i].n_cols);
			arma::uvec indeies = find(output[i]>rand_mat);
			dropoutMask[i] = arma::zeros(output[i].n_rows,output[i].n_cols);
			dropoutMask[i](indeies) = ones(indeies.size());
		}
	}
	
	

}
//return modelcost value
double ModelCost::modelbp(const arma::mat* features,arma::mat* dropoutMask,arma::mat *outputdelta,const int num_samples){
	arma::mat desired_out;
	double dropoutfraction = atof(params[layer_num].params[params_name[DROPOUTFRACTION]].c_str());
	double cost = 0;
	if(modules[layer_num-1]->name == "SoftMax"){

		desired_out = onehot(features[layer_num-1].n_rows,features[layer_num-1].n_cols,labels);
		cost += (weight_decay/2)*arma::accu(arma::pow(modules[layer_num-1]->weightMatrix,2));

	}
	else
		desired_out = labels.t();
	//arma::mat gm = reshape(desired_out,desired_out.size(),1).t()*log(reshape(activations[layer_num-1],activations[layer_num-1].size(),1));

	double gm_cost = dot(reshape(desired_out,desired_out.size(),1),log(reshape(features[layer_num-1],features[layer_num-1].size(),1)));

	cost += ((double)-1/num_samples)*gm_cost;
	
	
	
	//backward propagation to compute delta

	outputdelta[layer_num] = -(desired_out - features[layer_num-1]);
	arma::mat next_delta = outputdelta[layer_num];
	ofstream ofs;
	ofs.open("d_o.txt");
	outputdelta[layer_num].quiet_save(ofs,raw_ascii);
	ofs.close();

	for(int i = layer_num-1;i >=0 ;i--){
		
		outputdelta[i] = modules[i]->backpropagate(next_delta,features[i],params[i]);
		stringstream str_i;
		str_i << i;
		std::string deltaname = "delta_"+str_i.str() +".txt";
		ofs.open(deltaname);
		outputdelta[i].quiet_save(ofs,raw_ascii);
		ofs.close();

		if(i != layer_num-1 && dropoutfraction > 0){
			outputdelta[i] = outputdelta[i] % dropoutMask[i];
		}

		if(i > 0){
			next_delta = modules[i]->process_delta(outputdelta[i]);
		}
		

	}

	return cost;
}

double ModelCost::value_gradient(arma::mat& grad){

	double cost = 0;
	int num_samples = data.n_cols;
	arma::mat* activations = new arma::mat[layer_num];
	arma::mat *delta = new arma::mat[layer_num+1];
	
	arma::mat* dropoutMask = NULL;
	double dropoutfraction = atof(params[layer_num].params[params_name[DROPOUTFRACTION]].c_str());
	if(dropoutfraction > 0){
		dropoutMask = new arma::mat[layer_num];
	}


	grad = zeros(coefficient.size(),1);
	
	paramsToStack();


	//forward Propagation
	
	modelff(data,activations,dropoutMask);

	//backward propagation
	cost = modelbp(activations,dropoutMask,delta,num_samples);

	

	//compute gradient using delta
	ofstream ofs;
	
	int curr_loc = coefficient.size()-1;
	for(int i = layer_num-1;i >=0 ;i--){
		arma::mat w_grad = zeros(modules[i]->weightMatrix.n_rows,modules[i]->weightMatrix.n_cols);
		arma::mat b_grad = zeros(modules[i]->bias.size(),1);
		
		if(i == 0){
			modules[i]->calculate_grad_using_delta(data,delta[i],params[i],weight_decay,w_grad,b_grad);
			
		}
		else{
			modules[i]->calculate_grad_using_delta( activations[i-1],delta[i],params[i],weight_decay,w_grad,b_grad);
		}
		stringstream str_i;
		str_i << i;
		std::string wgradname = "wgrad_"+str_i.str() +".txt";
		std::string bgradname = "bgrad_"+str_i.str() +".txt";
		ofs.open(wgradname);
		w_grad.quiet_save(ofs,raw_ascii);
		ofs.close();
		ofs.open(bgradname);
		b_grad.quiet_save(ofs,raw_ascii);
		ofs.close();


		grad.rows(curr_loc - b_grad.size()+1, curr_loc) = reshape(b_grad,b_grad.size(),1);
		grad.rows(curr_loc-b_grad.size()-w_grad.size()+1,curr_loc-b_grad.size()) = reshape(w_grad,w_grad.size(),1);
		
		curr_loc = curr_loc - w_grad.size() - b_grad.size();
	}


	delete[] activations;
	delete[] delta;
	delete[] dropoutMask;
	return cost;
}
void ModelCost::gradient(arma::mat& grad){

}
void ModelCost::hessian(arma::mat& grad, arma::mat& hess){

}
void ModelCost::paramsToStack(){

	int curr_loc = 0;
	for(int i = 0;i < layer_num; i++){
		int hiddenSize = 0;
		int rows_num = 0,cols_num = 0;
		if(params[i].params[params_name[ALGORITHM]] == "ConvolveModule" ){
			int number_filters = ((ConvolveModule*) modules[i])->filterNum;
			int filter_dim = ((ConvolveModule*) modules[i])->filterDim;
			rows_num = filter_dim*number_filters;
			cols_num = filter_dim;

		}else if(params[i].params[params_name[ALGORITHM]] == "CRBM" ){
			int number_filters = ((ConvolutionRBM*) modules[i])->filterNum;
			int filter_dim = ((ConvolutionRBM*) modules[i])->filterDim;
			rows_num = filter_dim*number_filters;
			cols_num = filter_dim;

		}else if(params[i].params[params_name[ALGORITHM]] == "Pooling"){
			rows_num = ((Pooling*) modules[i])->outputImageNum;
			cols_num = 1;
		}else{
			rows_num = modules[i]->outputSize;
			cols_num = modules[i]->inputSize;
		}
		
		modules[i]->weightMatrix = arma::reshape(coefficient.rows(curr_loc,curr_loc + modules[i]->weightMatrix.size()-1),rows_num,cols_num);
		
		curr_loc += modules[i]->weightMatrix.size();

		arma::mat b = arma::reshape(coefficient.rows(curr_loc,curr_loc+modules[i]->bias.size()-1),modules[i]->bias.size(),1);
		modules[i]->bias = b;
		curr_loc += modules[i]->bias.size();
		
	}
}