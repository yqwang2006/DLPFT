#ifndef INTERPOINT_H
#define INTERPOINT_H
#include "complex.h"
class InterPoint{
public:
	Complex x;
	Complex f;
	Complex g;
	InterPoint(const double xx = 0,const double ff = 0, const double gg = 0):x(xx),f(ff),g(gg){} 
};


#endif