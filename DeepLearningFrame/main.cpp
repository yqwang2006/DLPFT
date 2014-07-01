#include "armadillo"
#include <string>
#include <time.h>
#include <iostream>
#include "io/LoadData.h"
#include "io/LoadParam.h"
#include "RegisterProduct.h"
#include "module/AutoEncoder.h"
#include "param/AllParam.h"
#include "io/AllDataAddr.h"
#include "model\Model.h"
#include "util\convolve.h"
#include "util\onehot.h"
using namespace std;
using namespace arma;
using namespace dlpft::factory;
using namespace dlpft::model;
using namespace dlpft::io;

//#define UNSUPERVISEDMODEL 1
void load_data(string ,arma::mat&);
void load_data(string ,arma::imat&);
void load_data(string ,arma::mat&, int, int);
void load_data(string ,arma::imat&,int, int);

int main(int argc, char**argv){

	if(argc < 3){
		exit(-1);
	}
	mat A = randu<mat>(4,7);
	cube B = randu<cube>(3,4,5);
	cout << A;
	A.cols(0,A.n_cols-2) = A.cols(1,A.n_cols-1);
	A.col(A.n_cols-1) = A.col(0);
	cout << A;
	string modelInfo = argv[1];
	string paramFileName = argv[2];
	string paramFullName = paramFileName + ".param";

	RegisterFunction();
	RegisterOptimizer();

	//load param file
	dlpft::io::LoadParam load_param(paramFullName);
	vector<vector<NewParam>> params;
	AllDataAddr data_addr;
	load_param.load(params,data_addr);
	
	arma::mat train_data,test_data,finetune_data;
	arma::imat train_labels,test_labels,finetune_labels;


	//load train data
	if(data_addr.train_data_addr == ""){
		cout << "Please set the address of the train data at the param file!" << endl;
		exit(-1);
	}
	cout << "Loading train data!" << endl;
	//load_data(data_addr.train_data_addr,train_data,18900,588);
	load_data(data_addr.train_data_addr,train_data,5000,784);
	train_data = train_data.t();
	if(data_addr.train_labels_addr != ""){
		//load_data(data_addr.train_labels_addr,train_labels,18900,1);
		load_data(data_addr.train_labels_addr,train_labels,5000,1);
	}
	

	//load test data, if test data == null, test labels must be null.
	if(data_addr.test_data_addr != ""){
		cout << "Loading test data!" << endl;
		//load_data(data_addr.test_data_addr,test_data,120617,588);
		load_data(data_addr.test_data_addr,test_data,1000,784);
		test_data = test_data.t();
		if(data_addr.test_labels_addr != ""){
			//load_data(data_addr.test_labels_addr,test_labels,120617,1);
			load_data(data_addr.test_labels_addr,test_labels,1000,1);		
		}
	}

	bool finetune_switch = false;

	if(data_addr.finetune_data_addr != "" && data_addr.finetune_labels_addr != ""){
		cout << "Loading finetune data!" << endl;
		load_data(data_addr.finetune_data_addr,finetune_data,950,588);
		finetune_data = finetune_data.t();
		load_data(data_addr.finetune_labels_addr,finetune_labels,950,1);
		finetune_switch = true;
	}else if(data_addr.train_labels_addr!=""){
		finetune_data = train_data;
		finetune_labels = train_labels;
		finetune_switch = true;
	}else{
		finetune_switch = false;
	}


	int input_size = train_data.n_rows;

	arma::imat pred_labels;
	double pred_acc = 0;

	clock_t start,end;
	double duration = 0;
	start = clock();

	cout << "Begin trainning!" << endl;
	if(modelInfo == "unSupervisedModel"){
		Model unsupervisedModel(input_size,params[0]);
		unsupervisedModel.pretrain(train_data,params[0]);
		if(finetune_switch){
			cout << "Begin finetuning!" << endl;
			unsupervisedModel.train_classifier(finetune_data,finetune_labels,params[0]);
			unsupervisedModel.train(finetune_data,finetune_labels,params[0]);
		}
		if(data_addr.test_data_addr != ""){
			cout << "Begin predicting!" << endl;
			pred_labels = unsupervisedModel.predict(test_data,test_labels,params[0]);
			if(data_addr.test_labels_addr != ""){
				pred_acc = unsupervisedModel.predict_acc(pred_labels,test_labels);
			}
		}
		
	}
	else{
		if(data_addr.train_labels_addr == ""){
			cout << "You must set the train labels! Because you choose the supervised model for train!" << endl;
			exit(-1);
		}
		Model supervisedModel(input_size,params[0]);
		supervisedModel.train(train_data,train_labels,params[0]);
		if(data_addr.test_data_addr != ""){
			cout << "Begin predicting!" << endl;
			pred_labels = supervisedModel.predict(test_data,test_labels,params[0]);
			if(data_addr.test_labels_addr != ""){
				pred_acc = supervisedModel.predict_acc(pred_labels,test_labels);
			}
		}
		
	}
	cout << "predict accu: " << pred_acc*100 << "%"<< endl;
	end = clock();
	duration = (double)(end-start)/CLOCKS_PER_SEC;
	cout << duration << endl;
	
	string result_file_name = "result/"+paramFileName + "_result.txt";
	ofstream ofs;
	ofs.open(result_file_name);
	ofs << pred_acc << endl;
	pred_labels.quiet_save(ofs,raw_ascii);
	ofs.close();




	return 0;
}

void load_data(string filename, arma::mat& data_mat,int rows,int cols){
	LoadData file(filename);
	clock_t start = clock(),end;
	double dur_time = 0;

	file.load_data_to_mat(data_mat,rows,cols);
	end = clock();
	dur_time = (double)(end-start)/CLOCKS_PER_SEC;
	cout << dur_time << endl;
	//cout << (*test)(998,716) << endl;
}
void load_data(string filename, arma::imat& data_mat,int rows,int cols){
	LoadData file(filename);
	clock_t start = clock(),end;
	double dur_time = 0;

	file.load_data_to_mat(data_mat,rows,cols);
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