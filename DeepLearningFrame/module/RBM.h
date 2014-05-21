#ifndef RBM_H
#define RBM_H
#include "Module.h"
namespace dlpft{
	namespace module{
		class RBM : public Module{
		public:
			arma::mat h_means;
			arma::mat h_samples;
			arma::mat nv_means;
			arma::mat nv_samples;
			arma::mat nh_means;
			arma::mat nh_samples;

			RBM() : Module(){}
			
			ResultModel run(arma::mat& data, arma::mat& labels, NewParam& param);
			~RBM();
		    void rand_data(arma::mat input, arma::mat* batches,int sample_num , int batch_size);
			void  CD_k(int k,arma::mat& input_data, arma::mat& weightMat, arma::mat& h_bias, arma::mat& v_bias);
			
			void sample_h_given_v(arma::mat& v0_sample, arma::mat& mean, arma::mat& sample,arma::mat& weightMat, arma::mat& h_bias);
		    void sample_v_given_h(arma::mat& h, arma::mat& v, arma::mat& sample,arma::mat& weightMat, arma::mat& v_bias);
		    arma::mat propup(arma::mat& v,arma::mat& weightMat, arma::mat& h_bias);
		    arma::mat propdown(arma::mat& h,arma::mat& weightMat, arma::mat& c_bias);
		    void  gibbs_hvh(arma::mat& weightMat, arma::mat& h_bias, arma::mat& v_bias,arma::mat& h0_sample);
		    double get_reconstruct_error(arma::mat& v);
		    arma::mat RBM_VtoH(arma::mat& input,ResultModel& result_model);
			arma::mat BiNomial(const arma::mat mean);
		};
	};
};

#endif