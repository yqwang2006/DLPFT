#ifndef SAVERESULT_H
#define SAVERESULT_H
#include "../module/Module.h"
#include <io.h>
#include <direct.h>
#include <mat.h>
#include <mex.h>
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
				
				for(int i = 0;i < params.size()-2;i++){
					/*string param_dir = dir_name + "\\result_param_" + getstring(i);
					if(_access(param_dir.c_str(),6) == -1){
						mkdir(param_dir.c_str());
					}*/
					string W_name = dir_name + "\\WeightMat_" + getstring(i) + ".mat";
					double *outA = new double[m[i]->weightMatrix.size()];  
					int M = m[i]->weightMatrix.n_rows;
					int N = m[i]->weightMatrix.n_cols;
					for (int k=0; k<M; k++)  
						for(int j = 0; j < N; j++)
							outA[M*j+k] = m[i]->weightMatrix(k,j);  
					MATFile *pmatFile = NULL;  
					mxArray *pMxArray = NULL;
					pmatFile = matOpen(W_name.c_str(),"w");  
					pMxArray = mxCreateDoubleMatrix(M, N, mxREAL);  
					mxSetData(pMxArray, outA);  
					matPutVariable(pmatFile, "weightMatrix", pMxArray);  
					matClose(pmatFile); 
					mxFree(outA);
				}
				if(params[params.size()-2].params[params_name[ALGORITHM]]=="SVM"){
					string W_name = dir_name + "\\svmmodel.txt";
					svm_save_model(W_name.c_str(),((SvmModule*)m[params.size()-1])->svmmodel); 
				}else{
					string W_name = dir_name + "\\WeightMat_" + getstring((int)(params.size()-2)) + ".mat";
					double *outA = new double[m[params.size()-2]->weightMatrix.size()];  
					int M = m[params.size()-2]->weightMatrix.n_rows;
					int N = m[params.size()-2]->weightMatrix.n_cols;
					for (int i=0; i<M; i++)  
						for(int j = 0; j < N; j++)
							outA[M*j+i] = m[i]->weightMatrix(i,j);  
					MATFile *pmatFile = NULL;  
					mxArray *pMxArray = NULL;
					pmatFile = matOpen(W_name.c_str(),"w");  
					pMxArray = mxCreateDoubleMatrix(M, N, mxREAL);  
					mxSetData(pMxArray, outA);  
					matPutVariable(pmatFile, "weightMatrix", pMxArray);  
					matClose(pmatFile); 
					mxFree(outA);


				}


			}
			void save_result(Module** m,vector<NewParam> params, string dir_name, mat pred_labels,string header_info){
				if(_access(dir_name.c_str(),6) == -1){
					mkdir(dir_name.c_str());
				}
				
				for(int i = 0;i < params.size()-2;i++){
					/*string param_dir = dir_name + "\\result_param_" + getstring(i);
					if(_access(param_dir.c_str(),6) == -1){
						mkdir(param_dir.c_str());
					}*/
					string W_name = dir_name + "\\WeightMat_" + getstring(i) + ".mat";
					string b_name = dir_name + "\\bias_" + getstring(i) + ".mat";
					MATFile *pmatFile = NULL;  
					mxArray *pMxArray = NULL;

					int M = m[i]->weightMatrix.n_rows;
					int N = m[i]->weightMatrix.n_cols;
					double *outA = new double[M*N];  
					for (int k=0; k<M; k++)  
						for(int j = 0; j < N; j++)
							outA[M*j+k] = m[i]->weightMatrix(k,j);  
					
					pmatFile = matOpen(W_name.c_str(),"w");  
					pMxArray = mxCreateDoubleMatrix(M, N, mxREAL);  
					mxSetData(pMxArray, outA);  
					matPutVariable(pmatFile, "weightMatrix", pMxArray);  

					/*pmatFile = matOpen(b_name.c_str(),"w");
					M = m[i]->bias.n_rows;
					N = m[i]->bias.n_cols;
					pMxArray = mxCreateDoubleMatrix(M,N,mxREAL);
					double *outB = new double[M*N];

					for (int k=0; k<M; k++)  
						for(int j = 0; j < N; j++)
							outB[M*j+k] = m[i]->weightMatrix(k,j);  

					mxSetData(pMxArray,outB);
					matPutVariable(pmatFile, "bias", pMxArray);*/
					//mxFree(outB);
					//matClose(pmatFile); 
					//mxFree(outA);
					//

				}
				if(params[params.size()-2].params[params_name[ALGORITHM]]=="SVM"){
					string W_name = dir_name + "\\svmmodel.txt";
					svm_save_model(W_name.c_str(),((SvmModule*)m[params.size()-2])->svmmodel); 
				}else{
					string W_name = dir_name + "\\WeightMat_" + getstring((int)(params.size()-2)) + ".mat";
					string b_name = dir_name + "\\bias_" + getstring((int)(params.size()-2)) + ".mat";

					double *outA = new double[m[params.size()-2]->weightMatrix.size()];  
					int M = m[params.size()-2]->weightMatrix.n_rows;
					int N = m[params.size()-2]->weightMatrix.n_cols;
					for (int k=0; k<M; k++)  
						for(int j = 0; j < N; j++)
							outA[M*j+k] = m[params.size()-2]->weightMatrix(k,j);  
					MATFile *pmatFile = NULL;  
					mxArray *pMxArray = NULL;
					pmatFile = matOpen(W_name.c_str(),"w");  
					pMxArray = mxCreateDoubleMatrix(M, N, mxREAL);  
					mxSetData(pMxArray, outA);  
					matPutVariable(pmatFile, "weightMatrix", pMxArray);  

					/*pmatFile = matOpen(b_name.c_str(),"w");
					M = m[params.size()-2]->bias.n_rows;
					N = m[params.size()-2]->bias.n_cols;
					pMxArray = mxCreateDoubleMatrix(M,N,mxREAL);
					double *outB = new double[M*N];

					for (int k=0; k<M; k++)  
						for(int j = 0; j < N; j++)
							outB[M*j+k] = m[params.size()-2]->weightMatrix(k,j);  

					mxSetData(pMxArray,outB);
					matPutVariable(pmatFile, "bias", pMxArray);
					
					mxFree(outB);*/

					matClose(pmatFile); 
					mxFree(outA);
					
				}


				string labels_name = dir_name+"\\predict_labels.mat";
				
				
				
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