#include "armadillo"
#include <string>
#include <time.h>
#include <iostream>
#include "io\LoadData.h"
#include "io\LoadParam.h"
#include "RegisterProduct.h"
#include "module\AllModule.h"
#include "param\AllParam.h"
#include "io\AllDataAddr.h"
#include "io\SaveResult.h"
#include "model\Model.h"
#include "util\convolve.h"
#include "util\onehot.h"
#include "util\global_vars.h"
#include "util\cuda_convolve.h"

using namespace std;
using namespace arma;
using namespace dlpft::factory;
using namespace dlpft::model;
using namespace dlpft::io;

//#define UNSUPERVISEDMODEL 1
void load_data(DataInfo ,arma::mat&);

SaveResult save_result;
int snap_num = 0;


int main(int argc, char**argv){
	//cube A= randn<cube> (3,2,2);
	//cout << A << endl;
	//cout << A(2) << ' ' << A(3) << endl;
	//A.reshape(6,2,1);
	//cout << A << endl;
	//mat B = A.slice(0);
	//cout << reshape(B,6,2) << endl;


	get_device_info();
	string paramFileFullPath;

	if(argc != 2){
		cout << "请输入参数文件路径和名称:" <<endl;
		getline(cin, paramFileFullPath);
	}else{
		paramFileFullPath = argv[1];
	}
	cout << paramFileFullPath << endl;
	string split_symbol;		//the split symbol in the full path of param file
	string paramFullName;		//the full name of param file, it includes ".param"
	string paramFullpath;		//the full path of param file. eg. if paramFullName = a/b/d.param,then paramFullpath = a/b/
	string paramFileName;		//the name of param file. paramFileName = paramFullName - paramFullPath - ".param"
	string filedir;				//the result file dir.

	/*parse the param file name received from argv*/

	if(paramFileFullPath.find(".param") != string::npos){
		paramFullName = paramFileFullPath;
	}else{
		paramFullName = paramFileFullPath + ".param";
	}


	if(paramFileFullPath.find("\\") != string::npos){
		split_symbol = "\\";
	}else {
		split_symbol = "/";
	}

	string::size_type pos = 0;
	if((pos = paramFullName.find_last_of(split_symbol)) != string::npos){
		paramFullpath = paramFullName.substr(0,pos);
	}else{
		paramFullpath = "";
	}
	paramFileName = paramFullName.substr(paramFullName.find_last_of(split_symbol)+1,paramFullName.find(".param")-paramFullName.find_last_of(split_symbol)-1);

	RegisterFunction();
	RegisterOptimizer();



	/*get the result file path to save the result files*/

	string param_path = "param";

	if((pos = paramFullpath.find(param_path)) != string::npos){
		filedir = paramFullpath .replace(pos,param_path.length(),"result" + split_symbol) + paramFileName;
	}else{
		filedir = paramFullpath + "result" + split_symbol + paramFileName;
	}
	string snapshotdir = "snapshot"+ split_symbol +paramFileName;
	if(_access(filedir.c_str(),6) == -1){
		mkdir(filedir.c_str());
	}

	open_file(filedir,"LogOut.txt");


	/*load param file from param file*/

	dlpft::io::LoadParam load_param(paramFullName);
	vector<vector<NewParam>> params;
	AllDataAddr data_addr;
	NewParam global_info;

	//params存放所有层的参数，其中params[layer_num]存放全局参数
	load_param.load(params,data_addr,global_info);
	arma::mat train_data,test_data,finetune_data;
	arma::mat train_labels,test_labels,finetune_labels;


	int layer_num = params[0].size();
	int class_number = atoi(params[0][layer_num-2].params[params_name[HIDNUM]].c_str());
	//load train data
	if(data_addr.train_data_info.name == ""){
		LogOut << "Please set the address of the train data at the param file!" << endl;
		cout << "Please set the address of the train data at the param file!" << endl;
		exit(-1);
	}

	LogOut << "Loading train data!" << endl;
	cout << "Loading train data!" << endl;

	load_data(data_addr.train_data_info,train_data);
	train_data = train_data.t();
	if(data_addr.train_labels_info.name != ""){
		load_data(data_addr.train_labels_info,train_labels);
		train_labels(find(train_labels==0))+=class_number;
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

	arma::mat pred_labels;
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

	//load test data, if test data == null, test labels must be null.
	if(data_addr.test_data_info.name != ""){

	}




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

	}
	else{
		if(data_addr.train_labels_info.name == ""){
			LogOut << "You must set the train labels! Because you choose the supervised model for train!" << endl;
			cout << "You must set the train labels! Because you choose the supervised model for train!" << endl;
			exit(-1);
		}
		model.train(train_data,train_labels,params[0]);


	}

	if(data_addr.test_data_info.name != ""){
		LogOut << "Loading test data!" << endl;
		cout << "Loading test data!" << endl;
		load_data(data_addr.test_data_info,test_data);
		test_data = test_data.t();
		if(data_addr.test_labels_info.name != ""){
			load_data(data_addr.test_labels_info,test_labels);

			test_labels(find(test_labels==0))+=class_number;

			LogOut << "Begin predicting!" << endl;
			cout << "Begin predicting!" << endl;

			pred_labels = model.predict(test_data,test_labels,params[0]);

			pred_acc = model.predict_acc(pred_labels,test_labels,class_number);

			LogOut << "predict accu: " << pred_acc*100 << "%"<< endl;
			cout << "predict accu: " << pred_acc*100 << "%"<< endl;
		}

	}


	end = clock();
	duration = (double)(end-start)/CLOCKS_PER_SEC;
	LogOut << duration << endl;


	string header_info = "Program consumed " + save_result.getstring(duration) + " s\n";
	header_info += "The accuracy is " + save_result.getstring(pred_acc*100) + "%\n";

	save_result.save_result(model.modules,params[0],filedir,pred_labels,header_info);


	close_file();

	return 0;
}

void load_data(DataInfo data, arma::mat& data_mat){
	LoadData file(data.name,data.var_name);
	clock_t start = clock(),end;
	double dur_time = 0;
	if(!file.load_data(data_mat)){
		file.load_data_to_mat(data_mat,data.rows,data.cols);
	}
	end = clock();
	dur_time = (double)(end-start)/CLOCKS_PER_SEC;
	cout << dur_time << endl;
	//cout << (*test)(998,716) << endl;
}

