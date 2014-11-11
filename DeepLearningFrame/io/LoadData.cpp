#include "LoadData.h"
#include <mat.h>
#include <mex.h>
#include "../util/global_vars.h"
bool dlpft::io::LoadData::load_data(arma::mat& data_mat){
	//extern float* initA;
	assert(file_name!="");
	bool unknow_type = false;
	size_t loc = file_name.rfind(".");
	if(loc == std::string::npos){
		std::cout << "Cannot determine type of file " << file_name << std::endl;
		LogOut << "Cannot determine type of file " << file_name << std::endl;
		return false;
	}
	std::fstream load_stream;
	std::string file_type_name = file_name.substr(loc+1);

	if(file_type_name == "mat"){
		MATFile *pmatFile = NULL;
		mxArray *pMxArray = NULL;

		// 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
		

		pmatFile = matOpen(file_name.c_str(),"r");
		pMxArray = matGetVariable(pmatFile, var_name.c_str());
		
		//std::cout <<file_name << ";"  <<var_name << std::endl;
		//initA = (float*) mxGetData(pMxArray);
		int M = mxGetM(pMxArray);
		int N = mxGetN(pMxArray);
		double* initA = (double *)mxGetData(pMxArray);
		data_mat.set_size(M,N);
		for (int i=0; i<M; i++){
			for (int j=0; j<N; j++){
				data_mat(i,j) = initA[M*j+i];
			}
		
		}
		matClose(pmatFile);
		mxFree(initA);
		return true;
	}

	load_stream.open(file_name.c_str(),std::fstream::in);
	if(!load_stream.is_open()){
		std::cout << "Cannot open the file! Please check file's name and try again!" << std::endl;
		LogOut << "Cannot open the file! Please check file's name and try again!" << std::endl;
		return false;
	}
	arma::file_type load_type;
	std::string string_type;
	//get type string
	
	//transform type string to lower case
	std::transform(file_type_name.begin(), file_type_name.end(), file_type_name.begin(),
		::tolower);
	if(file_type_name == "csv"){
		load_type = arma::csv_ascii;
		string_type = "CSV data";
	}else if(file_type_name == "txt"){
		const std::string ARMA_MAT_TXT = "ARMA_MAT_TXT";
		char* raw_header = new char[ARMA_MAT_TXT.length() + 1];
		std::streampos pos = load_stream.tellg();

		load_stream.read(raw_header, std::streamsize(ARMA_MAT_TXT.length()));
		raw_header[ARMA_MAT_TXT.length()] = '\0';
		load_stream.clear();
		load_stream.seekg(pos); 

		if (std::string(raw_header) == ARMA_MAT_TXT)
		{
			load_type = arma::arma_ascii;
			string_type = "Armadillo ASCII formatted data";
		}
		else
		{
			load_type = arma::raw_ascii;
			string_type = "raw ASCII formatted data";
				
		}
		delete[] raw_header;
	}else if(file_type_name == "pgm"){
		load_type = arma::pgm_binary;
		string_type = "PGM data";
	}else if (file_type_name == "bin"){
		const std::string ARMA_MAT_BIN = "ARMA_MAT_BIN";
		char *raw_header = new char[ARMA_MAT_BIN.length() + 1];

		std::streampos pos = load_stream.tellg();

		load_stream.read(raw_header, std::streamsize(ARMA_MAT_BIN.length()));
		raw_header[ARMA_MAT_BIN.length()] = '\0';
		load_stream.clear();
		load_stream.seekg(pos); 

		if (std::string(raw_header) == ARMA_MAT_BIN)
		{
			string_type = "Armadillo binary formatted data";
			load_type = arma::arma_binary;
		}
		else
		{
			string_type = "raw binary formatted data";
			load_type = arma::raw_binary;
		}

		delete[] raw_header;
	}else if(file_type_name == "mat"){
		load_type = arma::arma_ascii;
		string_type = "MAT data";

	}else{
		unknow_type = true;
		load_type = arma::raw_binary;
		string_type = "";
	}
	if(unknow_type){
		std::cout << "Cannot support " << file_type_name << " file type!" << std::endl;
		return false;
	}
	bool success = data_mat.load(load_stream,load_type);
	if(success){
		std::cout << "Loading from " << file_name << " successfully!" << std::endl;
		std::cout << "Size is " << (transpose ? data_mat.n_cols : data_mat.n_rows)
				  << " x " << (transpose ? data_mat.n_rows : data_mat.n_cols) << ".\n";
		LogOut << "Loading from " << file_name << " successfully!" << std::endl;
		LogOut << "Size is " << (transpose ? data_mat.n_cols : data_mat.n_rows)
				  << " x " << (transpose ? data_mat.n_rows : data_mat.n_cols) << ".\n";
	}
	if(transpose)
		data_mat = trans(data_mat);
	return success;
}

bool dlpft::io::LoadData::load_data_to_mat(arma::mat& data_mat,int rows,int cols){
	assert(file_name!="");
	size_t loc = file_name.rfind(".");
	if(loc == std::string::npos){
		std::cout << "Cannot determine type of file " << file_name << std::endl;
		LogOut << "Cannot determine type of file " << file_name << std::endl;
		return false;
	}
	if(rows == 0 || cols == 0){
		bool load_success = false;//load_data(data_mat);
		if(load_success) return true;
		else{
			getdatainfo(file_name,rows,cols);
		}
	}
	data_mat.set_size(rows,cols);
	double data_i_j = 0;
	std::ifstream ifs;
	ifs.open(file_name.c_str());
	if(!ifs.is_open()){
		std::cout << "Cannot open the file! Please check file's name and try again!" << std::endl;
		LogOut << "Cannot open the file! Please check file's name and try again!" << std::endl;
		return false;
	
	}
	for(int i = 0; i < rows;i++){
		for(int j = 0;j < cols;j++){
			ifs >> data_mat(i,j);
		}
	}
	return true;
}

void dlpft::io::LoadData::getdatainfo(std::string file_name,int& rows,int& cols){
	std::ifstream ifs;
	ifs.open(file_name);
	std::string line;
	int row_num = 0,col_num = 0;
	while(std::getline(ifs,line)){
		row_num ++;
		if(row_num == 1){
			std::string pattern ;
			if(line.find("\t")!=std::string::npos)
				pattern = "\t";
			else if(line.find(" ")!=std::string::npos)
				pattern = " ";
			else if(line.find(",")!=std::string::npos)
				pattern = ",";
			std::vector<std::string> words = split(line,pattern);
			cols = words.size();
		}
	}
	ifs.close();
	

	rows = row_num;
	
}
//字符串分割函数
std::vector<std::string> dlpft::io::LoadData::split(std::string str,std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str+=pattern;//扩展字符串以方便操作
    int size=str.size();
	int start_loc = str.find_first_not_of(pattern);
	int end_pos = 0;
    size_t i = start_loc;
	while(i < size){
		pos = str.find(pattern,i);
		start_loc = str.find_first_not_of(pattern,pos);
		if(pos < size){
			std::string s=str.substr(i,pos-i);
            result.push_back(s);
            i=start_loc;
		}

	}
	
    return result;
}