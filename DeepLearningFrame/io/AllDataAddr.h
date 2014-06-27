#ifndef ALLDATAADDR_H
#define ALLDATAADDR_H
#include <string>
using namespace std;
namespace dlpft{
	namespace io{
		class AllDataAddr{
		public:
			string train_data_addr;
			string train_labels_addr;
			string test_data_addr;
			string test_labels_addr;
			string finetune_data_addr;
			string finetune_labels_addr;
			AllDataAddr(){}
			~AllDataAddr(){}
		};
	};
};


#endif