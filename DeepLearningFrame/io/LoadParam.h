#pragma once
#include "armadillo"
#include <assert.h>
#include "../io/AllDataAddr.h"
#include "../param/AllParam.h"
#include "../param/NewParam.h"
namespace dlpft{
	namespace io{
		class LoadParam{
		private:
			std::string file_name;
			
		public:
			LoadParam(){file_name = "";}
			LoadParam(std::string fname):file_name(fname){}
			void setname(const string fileName){file_name = fileName;}
			void load(vector<vector<NewParam>>&, AllDataAddr&,NewParam&);
			std::vector<std::string> split(std::string str,std::string pattern);
		};
	};
};