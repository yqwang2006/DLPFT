#ifndef DATAINFO_H
#define DATAINFO_H
#include <string>
using namespace std;
namespace dlpft{
	namespace io{
		class DataInfo{
		public:
			string name;
			int rows;
			int cols;
			DataInfo(){
				name = "";
				rows = 0;
				cols = 0;
			}
		};
	};
};

#endif