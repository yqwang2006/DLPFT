#include "LoadParam.h"
#include "../util/params_name.h"
void dlpft::io::LoadParam::load(vector<vector<NewParam>>& result_vector, AllDataAddr& data_info,NewParam& globalInfo){
	
	//fill_param_map();
	ifstream infile;
	MultiParam* multi_params;
	char buf[1024];
	string line;
	infile.open(file_name);
	string algorithm;
	int layer_num = 0;
	int layer_order = 0;
	int iter = 0;
	int param_value_num = 1;
	int all_params = 1;
	
	if(infile.is_open()){
		while(infile.good() && !infile.eof()){
			memset(buf,0,1024);
			infile.getline(buf,1024);
			line = buf;
			if(line == "")
				continue;
			vector<string> words = split(line,":");
			string varname = words[0];
			string value = words[1];
			if(varname == params_name[LAYERNUM]){
				layer_num = atoi(value.c_str());
				multi_params = new MultiParam[layer_num+1];//last for global param
				continue;
			}
			if(varname == params_name[MODELTYPE]){
				globalInfo.params[params_name[MODELTYPE]] = value;
				layer_order = layer_num + 1;
				continue;
			}
			if(varname == params_name[FINETUNESWITCH]){
				globalInfo.params[params_name[FINETUNESWITCH]] = value;

				layer_order = layer_num + 1;
				
				continue;
			}
			if(varname == params_name[LAYERORDER]){
				layer_order = atoi(value.c_str());
				continue;
			}
			if(varname == params_name[ALGORITHM]){
				multi_params[layer_order-1].algorithm_name = varname;
				multi_params[layer_order-1].algorithm_value = value;
				iter = 0;
				continue;
			}

			vector<string> values = split(value,",");
			if(varname == params_name[TRAINDATA]){
				
				data_info.train_data_info.name = values[0].replace(value.find("%"),1,":");
				if(values.size() == 4){
					data_info.train_data_info.rows = atoi(values[2].c_str());
					data_info.train_data_info.cols = atoi(values[3].c_str());
				}else{
					data_info.train_data_info.rows = 0;
					data_info.train_data_info.cols = 0;
				}
				continue;

			}
			if(varname == params_name[TRAINLABELS]){
				data_info.train_labels_info.name = values[0].replace(value.find("%"),1,":");

				if(values.size() == 4){
					data_info.train_labels_info.rows = atoi(values[2].c_str());
					data_info.train_labels_info.cols = atoi(values[3].c_str());
				}else{
					data_info.train_data_info.rows = 0;
					data_info.train_data_info.cols = 0;
				}
				continue;
			}
			if(varname == params_name[TESTDATA]){
				data_info.test_data_info.name = values[0].replace(value.find("%"),1,":");
				if(values.size() == 4){
					data_info.test_data_info.rows = atoi(values[2].c_str());
					data_info.test_data_info.cols = atoi(values[3].c_str());
				}else{
					data_info.train_data_info.rows = 0;
					data_info.train_data_info.cols = 0;
				}
				continue;
			}
			if(varname == params_name[TESTLABELS]){
				data_info.test_labels_info.name = values[0].replace(value.find("%"),1,":");
				if(values.size() == 4){
					data_info.test_labels_info.rows = atoi(values[2].c_str());
					data_info.test_labels_info.cols = atoi(values[3].c_str());
				}else{
					data_info.train_data_info.rows = 0;
					data_info.train_data_info.cols = 0;
				}
				continue;
			}
			if(varname == params_name[FINETUNEDATA]){
				data_info.finetune_data_info.name = values[0].replace(value.find("%"),1,":");
				if(values.size() == 4){
					data_info.finetune_data_info.rows = atoi(values[2].c_str());
					data_info.finetune_data_info.cols = atoi(values[3].c_str());
				}else{
					data_info.train_data_info.rows = 0;
					data_info.train_data_info.cols = 0;
				}
				continue;
			}
			if(varname == params_name[FINETUNELABELS]){
				data_info.finetune_labels_info.name = values[0].replace(value.find("%"),1,":");
				if(values.size() == 4){
					data_info.finetune_labels_info.rows = atoi(values[2].c_str());
					data_info.finetune_labels_info.cols = atoi(values[3].c_str());
				}else{
					data_info.train_data_info.rows = 0;
					data_info.train_data_info.cols = 0;
				}
				continue;
			}
			ParamVar param_var;

			param_var.mode = (ParamMode)atoi(values[0].c_str());

			string mode = values[0];
			
			if(param_var.mode == VALUES){
				vector<string> var_values;

				for(int i = 1;i < values.size();i++){
					var_values.push_back(values[i].c_str());
				}

				param_var.values = var_values;
				param_var.index = 0;
				param_var.num = var_values.size();
				param_var.var_name = varname;
				param_var.acc_tag = 1;

				multi_params[layer_order-1].vars.push_back(param_var);
				
				param_value_num = param_value_num * var_values.size();
			}else if(param_var.mode == RANGE){
				assert(values.size()==3+1);//1means mode,3 means three values;
				vector<string> var_values;

				double minval = atof(values[0].c_str());
				double step = atof(values[1].c_str());
				double maxval = atof(values[2].c_str());

				for(double var = minval;var <= maxval;var += step){
					char var_char[128];
					sprintf(var_char,"%f",var);
					string var_s = var_char;
					var_values.push_back(var_s);
				}

				param_var.values = var_values;
				param_var.index = 0;
				param_var.num = var_values.size();
				param_var.var_name = varname;
				param_var.acc_tag = 1;
				multi_params[layer_order-1].vars.push_back(param_var);
				
				param_value_num = param_value_num * var_values.size();
			}else{
				
			}

			iter ++;
		}//end while
	}//end if open
	infile.close();

	//set acc_tag;
	all_params = param_value_num;
	for(int i = 0;i < layer_num+1;i++){
		for(int j = 0;j < multi_params[i].vars.size();j++){
			
				if(multi_params[i].vars[j].mode != AUTOSEARCH){
					param_value_num /= multi_params[i].vars[j].num;
					multi_params[i].vars[j].acc_tag = param_value_num;
				}
			
		}
	}

	int index = 0;
	for(int i = 0;i < all_params;i++){
		vector<NewParam> param_vec;
		for(int layer = 0;layer < layer_num + 1;layer ++){
			NewParam param_layer;
			param_layer.addNewParam(multi_params[layer].algorithm_name,multi_params[layer].algorithm_value);
			for(int k = 0;k < multi_params[layer].vars.size(); k++){
						if(multi_params[layer].vars[k].mode != AUTOSEARCH){
							index = multi_params[layer].vars[k].index;
							param_layer.addNewParam(multi_params[layer].vars[k].var_name,
															multi_params[layer].vars[k].values[index]);
							if((i+1) % multi_params[layer].vars[k].acc_tag == 0){
								multi_params[layer].vars[k].index ++;
								if(multi_params[layer].vars[k].index == multi_params[layer].vars[k].num){
									multi_params[layer].vars[k].index = multi_params[layer].vars[k].index - multi_params[layer].vars[k].num;
								}
							}

						}else{
							param_layer.addNewParam(multi_params[layer].vars[k].var_name,"-1");
						}
				
			}
			param_vec.push_back(param_layer);


		}
		result_vector.push_back(param_vec);
	
	}
	
	
	if(globalInfo.params[params_name[MODELTYPE]]== ""){
		globalInfo.params[params_name[MODELTYPE]] = "UnsuperviseModel";
	}
	if(globalInfo.params[params_name[FINETUNESWITCH]]== ""){
		globalInfo.params[params_name[FINETUNESWITCH]] = "OFF";
	}
	//for(int i = 0;i < all_params;i ++)
	//	for(int j = 0;j < layer_num+1;j++){
	//		cout << "Param["<< i << "]["<<j<<"] = " << endl;
	//		cout << result_vector[i][j];
	//	}

}

//×Ö·û´®·Ö¸îº¯Êý
std::vector<std::string> dlpft::io::LoadParam::split(std::string str,std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str+=pattern;//À©Õ¹×Ö·û´®ÒÔ·½±ã²Ù×÷
    int size=str.size();

    for(size_t i=0; i<size; i++)
    {
        pos=str.find(pattern,i);
        if(pos<size)
        {
            std::string s=str.substr(i,pos-i);
            result.push_back(s);
            i=pos+pattern.size()-1;
        }
    }
    return result;
}