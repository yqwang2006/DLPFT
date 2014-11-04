#pragma once
#include "armadillo"
#include <assert.h>
namespace dlpft{
	namespace io{
		
		class LoadData{
		private:
			std::string file_name;
			std::string var_name;//only for mat file
			bool transpose;
		public:
			LoadData(){file_name = "";transpose = false;}
			LoadData(const std::string fname, const std::string vname="",const bool transpose=false):file_name(fname),var_name(vname),transpose(transpose){}
			LoadData(int rs,int cl,const std::string fname, const bool transpose=false):file_name(fname),transpose(transpose){}
			bool load_data(arma::mat& data_mat);
			bool load_data_to_mat(arma::mat& data_mat,int rows,int cols);
			void getdatainfo(std::string file_name,int& rows,int& cols);
			std::vector<std::string> split(std::string str,std::string pattern);
		};
	};
};