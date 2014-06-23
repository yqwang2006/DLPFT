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
	string m_name = param.params[params_name[OPTIMETHOD]];
	int max_epoch = atoi(param.params[params_name[MAXEPOCH]].c_str());
	if(max_epoch == 0)
		max_epoch = 100;
				
	Optimizer* opt_ptr;
	if(m_name == "sgd"){
		int batch_size = atoi(param.params[params_name[BATCHSIZE]].c_str());
		double learning_rate = atof(param.params[params_name[LEARNRATE]].c_str());
		if(batch_size == 0) batch_size = 100;
		if(learning_rate == 0) learning_rate = 0.1;
		opt_ptr = new SgdOptimizer(cost_ptr,max_epoch,learning_rate,batch_size);
	}else if(m_name == "lbfgs"){
		opt_ptr = new LbfgsOptimizer(cost_ptr,max_epoch);
	}else{
		opt_ptr = new CgOptimizer(cost_ptr,max_epoch);
	}
	return opt_ptr;
}

#endif
