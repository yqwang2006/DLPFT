#ifndef SAVERESULT_H
#define SAVERESULT_H
#include "../module/Module.h"
#include <io.h>
#include <direct.h>
extern int snap_num;
namespace dlpft{
	namespace io{
		class SaveResult{
		public:
			SaveResult(){}
			string getstring(const int number){
				stringstream str;
				str << number;
				return str.str();
			}
			string getstring(const double number){
				stringstream str;
				str << number;
				return str.str();
			}
			void save_result(Module** m,vector<NewParam> params, string dir_name){
				if(_access(dir_name.c_str(),6) == -1){
					mkdir(dir_name.c_str());
				}
				
				for(int i = 0;i < params.size()-1;i++){
					/*string param_dir = dir_name + "\\result_param_" + getstring(i);
					if(_access(param_dir.c_str(),6) == -1){
						mkdir(param_dir.c_str());
					}*/
					string W_name = dir_name + "\\WeightMat_" + getstring(i) + ".txt";
					ofstream ofs;
					ofs.open(W_name);
					m[i]->weightMatrix.quiet_save(ofs,raw_ascii);
					ofs.close();
				}
			}
			void save_result(Module** m,vector<NewParam> params, string dir_name, mat pred_labels,string header_info){
				if(_access(dir_name.c_str(),6) == -1){
					mkdir(dir_name.c_str());
				}
				
				for(int i = 0;i < params.size()-1;i++){
					/*string param_dir = dir_name + "\\result_param_" + getstring(i);
					if(_access(param_dir.c_str(),6) == -1){
						mkdir(param_dir.c_str());
					}*/
					string W_name = dir_name + "\\WeightMat_" + getstring(i) + ".txt";
					ofstream ofs;
					ofs.open(W_name);
					m[i]->weightMatrix.quiet_save(ofs,raw_ascii);
					ofs.close();
					string b_name = dir_name + "\\bias_" + getstring(i) + ".txt";
					ofs.open(b_name);
					m[i]->bias.quiet_save(ofs,raw_ascii);
					ofs.close();
				}
				string labels_name = dir_name+"\\predict_labels.txt";
				ofstream ofs;
				ofs.open(labels_name);
				ofs << header_info << endl;
				pred_labels.quiet_save(ofs,raw_ascii);
				ofs.close();
			}
			void save_snapshot(Module** m,vector<NewParam> params, string dir_name, mat pred_labels,string header_info){
				if(_access(dir_name.c_str(),6) == -1){
					mkdir(dir_name.c_str());
				}
				
				for(int i = 0;i < params.size()-1;i++){
					/*string param_dir = dir_name + "\\result_param_" + getstring(i);
					if(_access(param_dir.c_str(),6) == -1){
						mkdir(param_dir.c_str());
					}*/
					string W_name = dir_name + "\\WeightMat_" + getstring(i) + ".txt";
					ofstream ofs;
					ofs.open(W_name);
					m[i]->weightMatrix.quiet_save(ofs,raw_ascii);
					ofs.close();
					string b_name = dir_name + "\\bias_" + getstring(i) + ".txt";
					ofs.open(b_name);
					m[i]->bias.quiet_save(ofs,raw_ascii);
					ofs.close();
				}
				string labels_name = dir_name+"\\predict_labels.txt";
				ofstream ofs;
				ofs.open(labels_name);
				ofs << header_info << endl;
				pred_labels.quiet_save(ofs,raw_ascii);
				ofs.close();
			}
		};
	};
};
extern SaveResult save_result;

#endif