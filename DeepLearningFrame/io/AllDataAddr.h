#ifndef ALLDATAADDR_H
#define ALLDATAADDR_H
#include "DataInfo.h"
namespace dlpft{
	namespace io{
		class AllDataAddr{
		public:
			DataInfo train_data_info;
			DataInfo train_labels_info;
			DataInfo test_data_info;
			DataInfo test_labels_info;
			DataInfo finetune_data_info;
			DataInfo finetune_labels_info;
			AllDataAddr(){}
			~AllDataAddr(){}
		};
	};
};


#endif