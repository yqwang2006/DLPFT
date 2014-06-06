#ifndef CNN_H
#define CNN_H
#include "Model.h"
using namespace dlpft::module;
namespace dlpft{
	namespace model{
		class CNN : public Model{
			public:
			CNN(){}
			~CNN(){}
			ResultModel* train(const arma::mat data,const arma::imat labels, vector<NewParam> param);
		};

	};
};


#endif