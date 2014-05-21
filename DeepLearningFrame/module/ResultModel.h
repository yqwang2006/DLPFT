#pragma once
#include "armadillo"
#include <string>
namespace dlpft{
	namespace module{
		class ResultModel{
		public:
			string algorithm_name;
			arma::mat features;
			arma::mat weightMatrix;
			arma::mat bias;

		};

	};
};