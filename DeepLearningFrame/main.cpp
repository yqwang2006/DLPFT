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
#include "model\UnsupervisedModel.h"
#include "model\CNN.h"
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

int main(int argc, char**argv){

	if(argc < 3){
		exit(-1);
	}



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
	
	arma::mat train_data,test_data;
	arma::imat train_labels,test_labels;


	//load train data
	load_data(data_addr.train_data_addr,train_data);
	train_data = train_data.t();
	load_data(data_addr.train_labels_addr,train_labels);

	//load test data
	load_data(data_addr.test_data_addr,test_data);
	test_data = test_data.t();
	load_data(data_addr.test_labels_addr,test_labels);


	int input_size = train_data.n_rows;

	arma::imat pred_labels;
	double pred_acc;

	if(modelInfo == "unSupervisedModel"){
		UnsupervisedModel unsupervisedModel(input_size,params[0]);
		unsupervisedModel.pretrain(train_data,train_labels,params[0]);
		pred_acc = unsupervisedModel.predict(test_data,test_labels,params[0],pred_labels);

	}
	else{
		CNN cnn(input_size,params[0]);
		cnn.train(train_data,train_labels,params[0]);
		pred_acc = cnn.predict(test_data,test_labels,params[0],pred_labels);
	}
	
	string result_file_name = "result/"+paramFileName + "_result.txt";
	ofstream ofs;
	ofs.open(result_file_name);
	ofs << pred_acc << endl;
	pred_labels.quiet_save(ofs,raw_ascii);
	ofs.close();


	return 0;
}

void test_matrix(){
	arma::mat m1(2,3);
	m1.randn();
	cout << m1;

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