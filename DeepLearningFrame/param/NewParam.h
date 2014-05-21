#pragma once
#include <map>
#include <string>
using namespace std;
namespace dlpft{
	namespace param{
		class NewParam{
		public:
			map<string,string> params;
			NewParam(){}
			~NewParam(){}
			void addNewParam(string n,string v){
				params[n] = v;
			}
			friend ostream& operator<< (ostream& out,NewParam & p){
				for(map<string,string>::iterator iter = p.params.begin();iter != p.params.end();iter ++){
					
					out << iter->first << ' ' << iter->second << endl;
				}
				return out;
			}
		};
	};
};