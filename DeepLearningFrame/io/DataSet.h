#ifndef DATASET_H
#define DATASET_H
#include "armadillo"
namespace dlpft{
	namespace io{
		class DataSet{
		public:
			arma::mat data;
			arma::mat labels;
			bool labels_not_null;
			DataSet(arma::mat d,arma::mat l,const bool labels_nn = true)
				:data(d),labels(l),labels_not_null(labels_nn){}
		};
	};
};


#endif