#include "AutoEncoder.h"

#include "../param/AEParam.h"
using namespace dlpft::param;
using namespace dlpft::module;
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
ResultModel AutoEncoder::run(arma::mat& data, arma::mat& labels, NewParam& param){
	
	ResultModel result_model;

	typedef Creator<Optimizer> OptFactory;
	OptFactory& opt_factory = OptFactory::Instance();

	

	int hid_size = atoi(param.params["Hid_num"].c_str());
	int vis_size = data.n_rows;
	int h_v_size = hid_size * vis_size;
	int sample_num = data.n_cols;

	SAECostFunction* costfunc = new SAECostFunction(vis_size,hid_size);
	arma::mat grad;
	cout << param.params["Optimize_method"] << endl;
	Optimizer* testOpt = opt_factory.createProduct(param.params["Optimize_method"]);

	costfunc->set_data(data);
	testOpt->set_func_ptr(costfunc);


	//cout << "before opt:" << costfunc->get_coefficient()->n_rows<<";" << costfunc->get_coefficient()->n_cols << endl;
	//cout << ((AEParam*)param)->get_max_epoch() << endl;
	testOpt->set_max_iteration(atoi(param.params["Max_epoch"].c_str()));
	testOpt->optimize("theta"); 

	//cout << "after opt:" << costfunc->get_coefficient()->n_rows<<";" << costfunc->get_coefficient()->n_cols << endl;
	
	result_model.algorithm_name = "AutoEncoder";
	result_model.weightMatrix = (costfunc->get_coefficient()).rows(0,h_v_size-1);
	result_model.weightMatrix.reshape(hid_size,vis_size);
	result_model.bias = (costfunc->get_coefficient()).rows(2*h_v_size,2*h_v_size+hid_size-1);

	arma::mat activation = result_model.weightMatrix * data + repmat(result_model.bias,1,data.n_cols);
	result_model.features = sigmoid(activation);
	return result_model;
}