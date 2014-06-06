#ifndef RANDPERM_H
#define RANDPERM_H
#include "armadillo"
#include <vector>


static std::vector<int>& randperm(int N)
{
	std::vector<int> rand_perm;
	for(int i =0;i < N;i++){
		rand_perm.push_back(i);
	}
	random_shuffle(rand_perm.begin(),rand_perm.end());
	
	return rand_perm;
}
#endif