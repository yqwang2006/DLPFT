#pragma once
#include "../optimizer/Optimizer.h"
#include "../optimizer/CgOptimizer.h"
#pragma once
#include "Param.h"

using namespace dlpft::optimizer;
namespace dlpft{
	namespace param{
	class AEParam : public Param{
	private:
		int visiable_size;
		int hidden_num;
		int max_epoch;
		string opt_method;
		string cost_function;
		double lambda;
		double sparsity;
		double beta;
	public:
		AEParam(){
			name = "SAE";
			visiable_size = 784;
			hidden_num = 400;
			max_epoch = 10;
			opt_method = "cg";
			cost_function = "saecost";
			lambda = 3e-3;
			beta = 3e-3;
			saa_mode = false;
		}
		AEParam(int v_size, int hid_num, int m_epoch, const string o_method = "cg", const string cost_f = "saecost",const double lam = 3e-3, const double be = 3e-3){
			name = "SAE";
			visiable_size = v_size;
			hidden_num = hid_num;
			max_epoch = m_epoch;
			opt_method = o_method;
			cost_function = cost_f;
			lambda = lam;
			beta = be;
			saa_mode = false;
		}
		int get_visiable_size() const{return visiable_size;}
		int get_hidden_num() const{return hidden_num;}
		int get_max_epoch() const {return max_epoch;}
		string get_opt_method() const{return opt_method;}
		string get_cost_function() const{return cost_function;}
		double get_lambda() const{return lambda;}
		double get_sparsity() const{return sparsity;}
		double get_beta() const{return beta;}

		void set_visiable_size(int v_size){ visiable_size = v_size;}
		void set_hidden_num(int hid_num){ hidden_num = hid_num;}
		void set_max_epoch(int max_iter){max_epoch = max_iter;}
		void set_opt_method(string opt_m){opt_method = opt_m;}
		void set_cost_function(string cost_func){cost_function = cost_func;}
		void set_lambda(double l){lambda = l;}
		void set_beta(double b){beta = b;}
		void set_sparsity(double s){sparsity = s;}

		AEParam & AEParam::operator=(const AEParam & p){
			hidden_num = p.hidden_num;
			max_epoch = p.max_epoch;
			opt_method = p.opt_method;
			cost_function = p.cost_function;
			lambda = p.lambda;
			beta = p.beta;
			return *this;
		}
	};
};
};