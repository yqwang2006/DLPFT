#ifndef GLOBALVARS_H
#define GLOBALVARS_H
#include<fstream>
#include <string>
#include <io.h>
#include <direct.h>
extern std::ofstream LogOut;
static bool open_file(std::string dir_name,std::string file_name){
	if(_access(dir_name.c_str(),6) == -1){
		mkdir(dir_name.c_str());
	}
	std::string file_path = dir_name + "\\" + file_name;
	LogOut.open(file_path);
	if(LogOut.is_open()){
		return true;
	}
	return false;
}
static void close_file(){
	LogOut.close();
}
#endif