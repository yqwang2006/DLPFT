#ifndef SCFEATURECOST_H
#define SCFEATURECOST_H

#include "CostFunction.h"
namespace dlpft{
	namespace function{
		class SCFeatureCost : public CostFunction{
		public:
			arma::mat weightMatrix;
			//now featureMat is coefficient
			int visible_size;
			int hidden_size;
			double lambda;
			double gamma;
			double epsilon;
			/* groupMatrix
			- the grouping matrix. groupMatrix(r, :) indicates the
             features included in the rth group. groupMatrix(r, c)
             is 1 if the cth feature is in the rth group and 0
             otherwise.*/
			bool is_topo;
			arma::mat group_matrix;

			SCFeatureCost():CostFunction(){}
			SCFeatureCost(int v, int h, arma::mat gm, double lam = 5e-5, const double gam = 1e-2, const double eps = 1e-5,const bool topo = false)
				:visible_size(v),hidden_size(h),group_matrix(gm),lambda(lam),gamma(gam),epsilon(eps),is_topo(topo){
				if(lambda == 0)
					lambda = 5e-5;
				if(gamma == 0)
					gamma = 1e-2;
				if(epsilon == 0)
					epsilon = 1e-5;
			}

			~SCFeatureCost(){}

			double value_gradient(arma::mat& grad);
			void gradient(arma::mat& grad);
			void hessian(arma::mat& grad, arma::mat& hess);

		};
	};
};

#endif