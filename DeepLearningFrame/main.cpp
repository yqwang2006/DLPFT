#include "armadillo"
#include <string>
#include <time.h>
#include <iostream>
#include "io/LoadData.h"
#include "io/LoadParam.h"
#include "RegisterProduct.h"
#include "module\AllModule.h"
#include "param/AllParam.h"
#include "io/AllDataAddr.h"
#include "io/SaveResult.h"
#include "model\Model.h"
#include "util\convolve.h"
#include "util\onehot.h"
#include "util\global_vars.h"
using namespace std;
using namespace arma;
using namespace dlpft::factory;
using namespace dlpft::model;
using namespace dlpft::io;

//#define UNSUPERVISEDMODEL 1
void load_data(string ,arma::mat&);
void load_data(string ,arma::imat&);
void load_data(DataInfo ,arma::mat&);
void load_data(DataInfo ,arma::imat&);
SaveResult save_result;
int snap_num = 0;
int main(int argc, char**argv){

	if(argc < 2){
		exit(-1);
	}
	string paramFileFullPath = argv[1];

	string paramFullName = paramFileFullPath + ".param";
	
	string paramFileName = paramFileFullPath.substr(paramFileFullPath.find_last_of("\\")+1,paramFileFullPath.length());


	RegisterFunction();
	RegisterOptimizer();

	string filedir = "result\\"+paramFileName;
	string snapshotdir = "snapshot\\"+paramFileName;


	open_file(filedir,"LogOut.txt");
	
	
	//load param file
	dlpft::io::LoadParam load_param(paramFullName);
	vector<vector<NewParam>> params;
	AllDataAddr data_addr;
	NewParam global_info;
	


	//params存放所有层的参数，其中params[layer_num]存放全局参数
	load_param.load(params,data_addr,global_info);
	arma::mat train_data,test_data,finetune_data;
	arma::imat train_labels,test_labels,finetune_labels;

	
	//load train data
	if(data_addr.train_data_info.name == ""){
		LogOut << "Please set the address of the train data at the param file!" << endl;
		cout << "Please set the address of the train data at the param file!" << endl;
		exit(-1);
	}
	LogOut << "Loading train data!" << endl;
	cout << "Loading train data!" << endl;
	//load_data(data_addr.train_data_addr,train_data,18900,588);
	load_data(data_addr.train_data_info,train_data);
	train_data = train_data.t();
	if(data_addr.train_labels_info.name != ""){
		//load_data(data_addr.train_labels_addr,train_labels,18900,1);
		load_data(data_addr.train_labels_info,train_labels);
	}
	

	//load test data, if test data == null, test labels must be null.
	if(data_addr.test_data_info.name != ""){
		LogOut << "Loading test data!" << endl;
		cout << "Loading test data!" << endl;
		//load_data(data_addr.test_data_addr,test_data,120617,588);
		load_data(data_addr.test_data_info,test_data);
		test_data = test_data.t();
		if(data_addr.test_labels_info.name != ""){
			//load_data(data_addr.test_labels_addr,test_labels,120617,1);
			load_data(data_addr.test_labels_info,test_labels);		
		}
	}

	bool finetune_data_switch = false;

	if(data_addr.finetune_data_info.name != "" && data_addr.finetune_labels_info.name != ""){
		LogOut << "Loading finetune data!" << endl;
		cout << "Loading finetune data!" << endl;
		load_data(data_addr.finetune_data_info,finetune_data);
		finetune_data = finetune_data.t();
		load_data(data_addr.finetune_labels_info,finetune_labels);
		finetune_data_switch = true;
	}else if(data_addr.train_labels_info.name!=""){
		finetune_data = train_data;
		finetune_labels = train_labels;
		finetune_data_switch = true;
	}else{
		finetune_data_switch = false;
	}


	int input_size = train_data.n_rows;

	arma::imat pred_labels;
	double pred_acc = 0;

	clock_t start,end;
	double duration = 0;
	start = clock();

	LogOut << "Begin trainning!" << endl;
	cout << "Begin trainning!" << endl;
	//只训练第一个参数文件，此处可加循环将所有的参数文件都训练一遍。
	
	string init_mat_from_file = global_info.params[params_name[LOADWEIGHT]];
	string file_path = global_info.params[params_name[WEIGHTADDRESS]];
	
	Model model(input_size,params[0],init_mat_from_file,file_path);
	int layer_num = params[0].size();
	if(global_info.params[params_name[MODELTYPE]] == "UnsuperviseModel"){
	
		model.pretrain(train_data,params[0]);
		if(data_addr.train_labels_info.name != ""){
			model.train_classifier(train_data,train_labels,params[0]);
		}
		if(finetune_data_switch && global_info.params[params_name[FINETUNESWITCH]] == "ON"){
			LogOut << "Begin finetuning!" << endl;
			cout << "Begin finetuning!" << endl;
			
			model.train(finetune_data,finetune_labels,params[0]);
		}
		if(data_addr.test_data_info.name != ""){
			LogOut << "Begin predicting!" << endl;
			cout << "Begin predicting!" << endl;
			pred_labels = model.predict(test_data,test_labels,params[0]);
			if(data_addr.test_labels_info.name != ""){
				pred_acc = model.predict_acc(pred_labels,test_labels);
			}
		}
		
	}
	else{
		if(data_addr.train_labels_info.name == ""){
			LogOut << "You must set the train labels! Because you choose the supervised model for train!" << endl;
			cout << "You must set the train labels! Because you choose the supervised model for train!" << endl;
			exit(-1);
		}
		model.train(train_data,train_labels,params[0]);
		if(data_addr.test_data_info.name != ""){
			LogOut << "Begin predicting!" << endl;
			cout << "Begin predicting!" << endl;
			pred_labels = model.predict(test_data,test_labels,params[0]);
			if(data_addr.test_labels_info.name != ""){
				pred_acc = model.predict_acc(pred_labels,test_labels);
			}
		}
		
	}
	
	LogOut << "predict accu: " << pred_acc*100 << "%"<< endl;
	cout << "predict accu: " << pred_acc*100 << "%"<< endl;
	end = clock();
	duration = (double)(end-start)/CLOCKS_PER_SEC;
	LogOut << duration << endl;
	

	string header_info = "Program consumed " + save_result.getstring(duration) + " s\n";
	header_info += "The accuracy is " + save_result.getstring(pred_acc*100) + "%\n";

	save_result.save_result(model.modules,params[0],filedir,pred_labels,header_info);

	
	close_file();

	return 0;
}

void load_data(DataInfo filename, arma::mat& data_mat){
	LoadData file(filename.name);
	clock_t start = clock(),end;
	double dur_time = 0;
	if(!file.load_data(data_mat)){
		file.load_data_to_mat(data_mat,filename.rows,filename.cols);
	}
	end = clock();
	dur_time = (double)(end-start)/CLOCKS_PER_SEC;
	cout << dur_time << endl;
	//cout << (*test)(998,716) << endl;
}
void load_data(DataInfo filename, arma::imat& data_mat){
	LoadData file(filename.name);
	clock_t start = clock(),end;
	double dur_time = 0;

	if(!file.load_data(data_mat)){
		file.load_data_to_mat(data_mat,filename.rows,filename.cols);
	}
	end = clock();
	dur_time = (double)(end-start)/CLOCKS_PER_SEC;
	cout << dur_time << endl;
	//cout << (*test)(998,716) << endl;
}
void load_data(string filename, arma::mat& data_mat){
	LoadData file(filename);
	clock_t start = clock(),end;
	double dur_time = 0;

	file.load_data(data_mat);
	end = clock();
	dur_time = (double)(end-start)/CLOCKS_PER_SEC;
	cout << dur_time << endl;
	//cout << (*test)(998,716) << endl;
}
void load_data(string filename, arma::imat& data_mat){
	LoadData file(filename);
	clock_t start = clock(),end;
	double dur_time = 0;

	file.load_data(data_mat);
	end = clock();
	dur_time = (double)(end-start)/CLOCKS_PER_SEC;
	cout << dur_time << endl;
	//cout << (*test)(998,716) << endl;
}