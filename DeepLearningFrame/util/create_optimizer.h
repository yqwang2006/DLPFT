#ifndef CREATE_OPTIMIZER_H
#define CREATE_OPTIMIZER_H
#include "../optimizer/AllOptMethod.h"
#include "../param/NewParam.h"
#include "../function/CostFunction.h"
#include "params_name.h"
using namespace dlpft::param;
using namespace dlpft::optimizer;
using namespace dlpft::function;
static Optimizer* create_optimizer(NewParam& param,CostFunction* cost_ptr){
	
	int max_epoch = 100;
	double learning_rate = 0.1;
	double learning_rate_decay = 0.98;
	int batch_size = 100;

	string m_name = param.params[params_name[OPTIMETHOD]];
	if(m_name == ""){
		m_name = param.params[params_name[GLOBALOPTMETHOD]];
		max_epoch = atoi(param.params[params_name[GLOBALMAXEPOCH]].c_str());
		learning_rate = atof(param.params[params_name[GLOBALLEARNRATE]].c_str());
		learning_rate_decay = atof(param.params[params_name[GLOBALLEARNRATEDECAY]].c_str());
		batch_size = atof(param.params[params_name[GLOBALBATCHSIZE]].c_str());

	}else{
		max_epoch = atoi(param.params[params_name[MAXEPOCH]].c_str());
		learning_rate = atof(param.params[params_name[LEARNRATE]].c_str());
		batch_size = atof(param.params[params_name[BATCHSIZE]].c_str());
	}

	if(max_epoch == 0)
		max_epoch = 100;
	if(learning_rate == 0)
		learning_rate = 0.1;
	if(learning_rate_decay == 0)
		learning_rate_decay = 0.98;
	if(batch_size == 0)
		batch_size = 100;


	Optimizer* opt_ptr;
	if(m_name == "sgd"){
		opt_ptr = new SgdOptimizer(cost_ptr,max_epoch,learning_rate,batch_size,learning_rate_decay);
	}else if(m_name == "cg"){
		opt_ptr = new CgOptimizer(cost_ptr,max_epoch);
	}else{
		opt_ptr = new LbfgsOptimizer(cost_ptr,max_epoch);
	}
	return opt_ptr;
}

#endif
