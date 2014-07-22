#ifndef MULTIPARAM_H
#define MULTIPARAM_H

#include <vector>
namespace dlpft{
	namespace param{
		enum ParamMode {VALUES=1,RANGE,AUTOSEARCH};
		class ParamVar{
		public:
			std::string var_name;
			std::vector<string> values;
			ParamMode mode;
			int acc_tag;
			int num;
			int index;
			ParamVar(){}
			ParamVar(string name, vector<string> value, ParamMode m, int acc, int n, int ind)
					:var_name(name),values(value),mode(m),acc_tag(acc),num(n),index(ind){}
			~ParamVar(){}

		};
		class MultiParam{
		public:
			std::string algorithm;
			std::vector<ParamVar> vars;

		};
	};

};


#endif