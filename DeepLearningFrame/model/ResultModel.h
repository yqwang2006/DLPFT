#pragma once
#include "armadillo"
#include <string>
namespace dlpft{
	namespace model{
		class ResultModel{
		public:
			string algorithm_name;
			arma::mat weightMatrix;
			arma::mat bias;
			ResultModel& operator=(const ResultModel & c){
				algorithm_name = c.algorithm_name;
				weightMatrix = c.weightMatrix;
				bias = c.bias;
				return *this;
			}
		};

	};
};