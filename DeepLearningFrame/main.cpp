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
using namespace std;
using namespace arma;
using namespace dlpft::factory;
using namespace dlpft::model;
using namespace dlpft::io;

#define UNSUPERVISEDMODEL 1

void load_data(string ,arma::mat&);
void load_data(string ,arma::imat&);

int main(){

	RegisterFunction();
	RegisterOptimizer();
	//mat image = zeros(16,5);
	//
	//for(int i=0;i<image.n_cols;i++){
	//	image.col(i) = (1+i)*ones(16,1);
	//}
	//cout << image;

	//cube images = arma::zeros(16,5,1);
	//images.slice(0) = image;
	//images.reshape(4,4,5);
	//cout << images;

	//cube patch = images.tube(1,1,2,2);
	//cout << images;
	//cout << patch;

	//mat weight = 0.7*ones(2,2);
	//cout << weight;

	//
	//cout << convn_cube(images,weight,"full");



#if UNSUPERVISEDMODEL
	dlpft::io::LoadParam load_param("CRBM.param");
#else
	dlpft::io::LoadParam load_param("CNN1.param");
#endif
	vector<vector<NewParam>> params;
	AllDataAddr data_addr;
	load_param.load(params,data_addr);
	

	arma::mat train_data;
	arma::imat train_labels;

	load_data(data_addr.train_data_addr,train_data);
	train_data = train_data.t();
	load_data(data_addr.train_labels_addr,train_labels);
	
	
	int input_size = train_data.n_rows;

#if UNSUPERVISEDMODEL
		UnsupervisedModel unsupervisedModel(input_size,params[0]);
		unsupervisedModel.pretrain(train_data,train_labels,params[0]);
	
#else
		CNN cnn(input_size,params[0]);
		cnn.train(train_data,train_labels,params[0]);
	
#endif
	arma::mat test_data;
	arma::imat test_labels;

	load_data(data_addr.test_data_addr,test_data);
	test_data = test_data.t();
	load_data(data_addr.test_labels_addr,test_labels);

#if UNSUPERVISEDMODEL 
		unsupervisedModel.predict(test_data,test_labels,params[0]);
	
#else
		cnn.predict(test_data,test_labels,params[0]);
	

#endif
	//


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