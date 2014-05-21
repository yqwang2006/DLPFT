#pragma once
#include "../optimizer/Optimizer.h"
#include "../optimizer/CgOptimizer.h"
#pragma once
#include "Param.h"

using namespace dlpft::optimizer;
namespace dlpft{
	namespace param{
	class SMParam : public Param{
	private:
		int visiable_size;
		int number_classes;
		int max_epoch;
		string opt_method;
		string cost_function;
	public:
		SMParam(){
			name = "SoftMax";
			max_epoch = 400;
			opt_method = "cg";
			cost_function = "softmaxcost";
			saa_mode = false;
		}
		SMParam(int v_size, int n_classes, int max_epoch, const string o_method = "cg", const string cost_f = "saecost",const double lam = 3e-3, const double be = 3e-3){
			name = "SoftMax";
			visiable_size = v_size;
			number_classes = n_classes;
			max_epoch = max_epoch;
			opt_method = o_method;
			cost_function = cost_f;
			saa_mode = false;
		}

		int get_visiable_size() const{return visiable_size;}
		int get_number_classes() const{return number_classes;}
		int get_max_epoch() const {return max_epoch;}
		string get_opt_method() const{return opt_method;}
		string get_cost_function() const{return cost_function;}


		void set_visiable_size(int v_size){visiable_size = v_size;}
		void set_number_classes(int n_classes){number_classes = n_classes;}
		void set_max_epoch(int max_iter){max_epoch = max_iter;}
		void set_opt_method(string opt_m){opt_method = opt_m;}
		void set_cost_function(string cost_func){cost_function = cost_func;}

		SMParam & SMParam::operator=(const SMParam & p){
			visiable_size = p.visiable_size;
			number_classes = p.number_classes;
			max_epoch = p.max_epoch;
			opt_method = p.opt_method;
			cost_function = p.cost_function;
			return *this;
		}
	};
};
};