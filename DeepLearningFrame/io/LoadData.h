#pragma once
#include "armadillo"
#include <assert.h>
namespace dlpft{
	namespace io{
		
		class LoadData{
		private:
			std::string file_name;
			bool transpose;
			
		public:
			LoadData(){file_name = "";transpose = false;}
			LoadData(const std::string fname, const bool transpose=false):file_name(fname),transpose(transpose){}
			bool load_data(arma::mat& data_mat);
		};
	};
};