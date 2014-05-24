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
#include "model\TrainModel.h"
#include "model\PredictModel.h"
using namespace std;
using namespace arma;
using namespace dlpft::factory;
using namespace dlpft::model;
using namespace dlpft::io;
arma::mat load_data(string filename);

int main(){

	RegisterFunction();
	RegisterOptimizer();


	dlpft::io::LoadParam load_param("AE.param");
	vector<vector<NewParam>> params;
	AllDataAddr data_addr;
	load_param.load(params,data_addr);


	arma::mat train_data = load_data(data_addr.train_data_addr);
	train_data = train_data.t();
	
	arma::mat train_labels = load_data(data_addr.train_labels_addr);
	

	TrainModel trainmodel(train_data,train_labels);
	ResultModel* resultmodel_ptr = trainmodel.pretrain(train_data,train_labels,params[0]);

	arma::mat test_data = load_data(data_addr.test_data_addr);
	arma::mat test_labels = load_data(data_addr.test_labels_addr);
	test_data = test_data.t();
	PredictModel testmodel(test_data,test_labels,params[0]);
	testmodel.predict(resultmodel_ptr,test_data,test_labels,params[0]);


	
	return 0;
}

void test_matrix(){
	arma::mat m1(2,3);
	m1.randn();
	cout << m1;

}
arma::mat load_data(string filename){
	LoadData file(filename);
	clock_t start = clock(),end;
	double dur_time = 0;
	arma::mat data_mat;

	file.load_data(data_mat);
	end = clock();
	dur_time = (double)(end-start)/CLOCKS_PER_SEC;
	cout << dur_time << endl;
	//cout << (*test)(998,716) << endl;
	return data_mat;
}
