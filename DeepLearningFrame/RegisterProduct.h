#pragma once

#include "optimizer\AllOptMethod.h"
#include "function\AllFunction.h"
#include "module\AllModule.h"
using namespace dlpft::function;
using namespace dlpft::optimizer;
using namespace dlpft::factory;
static void RegisterFunction(){
	/*添加CostFunction到工厂中*/
	typedef Creator<CostFunction> FuncCreator;
	FuncCreator& func_factory = FuncCreator::Instance();
	func_factory.registerCreator<SAECostFunction>("saecost");
	func_factory.registerCreator<ScWeightFunction>("scweightcost");
	func_factory.registerCreator<SoftMaxCost>("softmaxcost");
	func_factory.registerCreator<TestFunction>("test");
}
static void RegisterOptimizer(){

	/*添加Optimizer类到工厂中*/
	typedef Creator<Optimizer> OptCreator;
	OptCreator& opt_factory = OptCreator::Instance();
	opt_factory.registerCreator<CgOptimizer>("cg");
	opt_factory.registerCreator<SgdOptimizer>("sgd");
	opt_factory.registerCreator<LbfgsOptimizer>("lbfgs");

}