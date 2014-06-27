#include "LoadData.h"
bool dlpft::io::LoadData::load_data(arma::mat& data_mat){
	assert(file_name!="");
	bool unknow_type = false;
	size_t loc = file_name.rfind(".");
	if(loc == std::string::npos){
		std::cout << "Cannot determine type of file " << file_name << std::endl;
		return false;
	}
	std::fstream load_stream;

	load_stream.open(file_name.c_str(),std::fstream::in);
	if(!load_stream.is_open()){
		std::cout << "Cannot open the file! Please check file's name and try again!" << std::endl;
		return false;
	}
	arma::file_type load_type;
	std::string string_type;
	//get type string
	std::string file_type_name = file_name.substr(loc+1);
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
			load_type = arma::diskio::guess_file_type(load_stream);

			if (load_type == arma::raw_ascii) 
				string_type = "raw ASCII formatted data";
			else if (load_type == arma::csv_ascii)
				string_type = "CSV data";
			else 
				string_type = "raw ASCII formatted data";
				//unknow_type = true;
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
	if(!success){
		std::cout << std::endl;
		std::cout << "Loading from " << file_name << " failed!" << std::endl;
	}else{
		std::cout << "Loading from " << file_name << " successfully!" << std::endl;
		std::cout << "Size is " << (transpose ? data_mat.n_cols : data_mat.n_rows)
				  << " x " << (transpose ? data_mat.n_rows : data_mat.n_cols) << ".\n";
	}
	if(transpose)
		data_mat = trans(data_mat);
	return success;
}
bool dlpft::io::LoadData::load_data(arma::imat& data_mat){
	assert(file_name!="");
	bool unknow_type = false;
	size_t loc = file_name.rfind(".");
	if(loc == std::string::npos){
		std::cout << "Cannot determine type of file " << file_name << std::endl;
		return false;
	}
	std::fstream load_stream;

	load_stream.open(file_name.c_str(),std::fstream::in);
	if(!load_stream.is_open()){
		std::cout << "Cannot open the file! Please check file's name and try again!" << std::endl;
		return false;
	}
	arma::file_type load_type;
	std::string string_type;
	//get type string
	std::string file_type_name = file_name.substr(loc+1);
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
			load_type = arma::diskio::guess_file_type(load_stream);

			if (load_type == arma::raw_ascii) 
				string_type = "raw ASCII formatted data";
			else if (load_type == arma::csv_ascii)
				string_type = "CSV data";
			else 
				unknow_type = true;
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
	if(!success){
		std::cout << std::endl;
		std::cout << "Loading from " << file_name << " failed!" << std::endl;
	}else{
		std::cout << "Loading from " << file_name << " successfully!" << std::endl;
		std::cout << "Size is " << (transpose ? data_mat.n_cols : data_mat.n_rows)
				  << " x " << (transpose ? data_mat.n_rows : data_mat.n_cols) << ".\n";
	}
	if(transpose)
		data_mat = trans(data_mat);
	return success;
}
bool dlpft::io::LoadData::load_data_to_mat(arma::mat& data_mat,int rows,int cols){
	assert(file_name!="");
	bool unknow_type = false;
	size_t loc = file_name.rfind(".");
	if(loc == std::string::npos){
		std::cout << "Cannot determine type of file " << file_name << std::endl;
		return false;
	}
	data_mat.set_size(rows,cols);
	double data_i_j = 0;
	std::ifstream ifs;
	ifs.open(file_name.c_str());
	for(int i = 0; i < rows;i++){
		for(int j = 0;j < cols;j++){
			ifs >> data_i_j;
			data_mat(i,j) = data_i_j;
		}
	}
	return true;
}
bool dlpft::io::LoadData::load_data_to_mat(arma::imat& data_mat,int rows,int cols){
	assert(file_name!="");
	bool unknow_type = false;
	size_t loc = file_name.rfind(".");
	if(loc == std::string::npos){
		std::cout << "Cannot determine type of file " << file_name << std::endl;
		return false;
	}
	data_mat.set_size(rows,cols);
	std::ifstream ifs;
	int label;
	ifs.open(file_name.c_str());
	for(int i = 0; i < rows;i++){
		for(int j = 0;j < cols;j++){
			ifs >> label;
			data_mat(i,j) = label;
		}
	}
	ifs.close();
	return true;
}