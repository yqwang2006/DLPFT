#ifndef COMPLEX_H
#define COMPLEX_H
#include <iostream>
using namespace std;
class Complex{
public:
	double real;
	double imag;
	Complex(){real = 0;imag = 0;}
	Complex(double r):real(r),imag(0){}
	Complex(double r,double i):real(r),imag(i){}
	bool isReal(){return (imag == 0);}

	Complex & Complex::operator=(const Complex & c){
		real = c.real;
		imag = c.imag;
		return *this;
	}
	Complex & Complex::operator=(const double & c){
		real = c;
		imag = 0;
		return *this;
	}
	friend ostream& operator<< (ostream& out,Complex & c){
		out << ' ' << c.real << "+ i" << c.imag << ' ';

		return out;
	}
};

#endif