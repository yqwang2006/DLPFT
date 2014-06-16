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
using namespace std;
using namespace arma;
using namespace dlpft::factory;
using namespace dlpft::model;
using namespace dlpft::io;

void load_data(string ,arma::mat&);
void load_data(string ,arma::imat&);

int main(){

	RegisterFunction();
	RegisterOptimizer();


	dlpft::io::LoadParam load_param("CNN1.param");
	vector<vector<NewParam>> params;
	AllDataAddr data_addr;
	load_param.load(params,data_addr);
	

	arma::mat train_data;
	arma::imat train_labels;

	load_data(data_addr.train_data_addr,train_data);
	train_data = train_data.t();
	load_data(data_addr.train_labels_addr,train_labels);
	
	
	int input_size = train_data.n_rows;

	//UnsupervisedModel unsupervisedModel(input_size,params[0]);
	//unsupervisedModel.pretrain(train_data,train_labels,params[0]);

	CNN cnn(input_size,params[0]);
	cnn.train(train_data,train_labels,params[0]);

	arma::mat test_data;
	arma::imat test_labels;

	load_data(data_addr.test_data_addr,test_data);
	test_data = test_data.t();
	load_data(data_addr.test_labels_addr,test_labels);

	
	cnn.predict(test_data,test_labels,params[0]);
	//unsupervisedModel.predict(test_data,test_labels,params[0]);


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