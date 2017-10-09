// support functions for an adaptive linear combiner neural network
#ifndef GLOBALFUNCTIONS_H
#define GLOBALFUNCTIONS_H

#include<vector>
#include<cmath>
#include<cstdlib>

using namespace std;

// bug in win7 QT Creator
#ifndef M_PI
#define M_PI 3.141592653589793
#endif

// transfer functions for use with neural networks
void hardLimit      (const vector<double> &input, vector<double> &output);
void symHardLimit   (const vector<double> &input, vector<double> &output);
void pureLin        (const vector<double> &input, vector<double> &output);
void posLin         (const vector<double> &input, vector<double> &output);
void satLin         (const vector<double> &input, vector<double> &output);
void logSig         (const vector<double> &input, vector<double> &output);
void symLogSig      (const vector<double> &input, vector<double> &output);

// matrix support functions - symbols follow neural net use
// multiply a rectangular matrix and a vector
void vectorMultiply(const vector<vector<double> > &W,
                    const vector<double> &p, vector<double> &out);
// multiply two vectors
void vectorMultiply(const vector<double> &W,
                    const vector<double> &p, double &out);

/*
double Deg2Rad(const double deg)
{ return deg * M_PI / 180.0; }

double Rad2Deg(const double rad)
{ return rad * 180.0 / M_PI; }
*/

template <class T>
T Deg2Rad(const T& deg)
{ return T(deg * M_PI / 180.0); }

template <class T>
T Rad2Deg(const T& rad)
{ return T(rad * 180.0 / M_PI); }


// scale, normalize or find max abs value of vector
void Scale(vector<double> &vect, const double scale = 1.0);
void Normalize(vector<double> &vect);
double MaxAbsVal(const vector<double> &vect);

#endif // GLOBALFUNCTIONS_H

