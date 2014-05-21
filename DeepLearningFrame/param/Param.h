#pragma once
namespace dlpft{
	namespace param{
		enum ParamMode {VALUES=1,RANGE,AUTOSEARCH};
		class Param{
		public:
			string name;
			bool saa_mode;
			Param(){saa_mode = false;}
			Param(string n, const bool saa = false):name(n),saa_mode(saa){}
			~Param(){}

		};
	};
};