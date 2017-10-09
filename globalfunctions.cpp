// support functions for an adaptive linear combiner neural network
#include "globalfunctions.h"

#include<vector>
#include<cmath>
#include<exception>
#include<stdexcept>

using namespace std;

/*=============================================================================
    hardLimit
=============================================================================*/
void hardLimit(const vector<double> &input, vector<double> &output)
{
    output.clear();

    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        if (input[idx] >= 0.0)
            output.push_back(1.0);
        else
            output.push_back(0.0);
    }
}

/*=============================================================================
    symHardLimit
=============================================================================*/
void symHardLimit(const vector<double> &input, vector<double> &output)
{
    output.clear();

    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        if (input[idx] >= 0.0)
            output.push_back(1.0);
        else
            output.push_back(-1.0);
    }
}

/*=============================================================================
    pureLin - simply copies the input to the output
=============================================================================*/
void pureLin(const vector<double> &input, vector<double> &output)
{
    output.clear();
    output.assign(input.begin(), input.end());
}

/*=============================================================================
    posLin
=============================================================================*/
void posLin(const vector<double> &input, vector<double> &output)
{
    output.clear();

    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        if (input[idx] > 0.0)
            output.push_back(input[idx]);
        else
            output.push_back(0.0);
    }
}

/*=============================================================================
    satLin
=============================================================================*/
void satLin (const vector<double> &input, vector<double> &output)
{
    output.clear();

    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        if (input[idx] > 1.0)
            output.push_back(1.0);
        else if (input[idx] < -1.0)
            output.push_back(-1.0);
        else
            output.push_back(input[idx]);
    }
}

/*=============================================================================
    logSig - time consuming
=============================================================================*/
void logSig(const vector<double> &input, vector<double> &output)
{
    output.clear();

    for (unsigned int i = 0; i < input.size(); ++i)
        output.push_back(1.0 / (1.0 + exp(-input[i])));
}

/*=============================================================================
    symLogSig - time consuming
    logSig expanded to -1:1
=============================================================================*/
void symLogSig (const vector<double> &input, vector<double> &output)
{
    output.clear();

    for (unsigned int i = 0; i < input.size(); ++i)
    {
        output.push_back(2.0 / (1.0 + exp(-input[i])) - 1.0);

        if (output[i] > 1.000001 || output[i] < -1.000001)
            output[i] -= .002;
    }
}

/*=============================================================================
   vectorMultiply (could also be done as operator override)
      This version multiplys a vector by a rectangular matrix
=============================================================================*/
void vectorMultiply(const vector<vector<double> > &W,
                    const vector<double> &p, vector<double> &out)
{
   out.clear();

   // throwing exceptions here rather than returning false means that
   // the error can be handled up in the gui layer without cluttering up the
   // callers with 'if' statements
   if (p.size() == 0)
      throw std::invalid_argument("Zero vector size in vectorMultiply");

   // check that w has rows
   if (W.size() == 0)
      throw std::invalid_argument("Rectangular array size = 0 vectorMultiply");

   const double P_COLS = p.size();
   unsigned int row, col;
   #define VECT_IDX col
   double rowVal;

   for (row = 0; row < W.size(); ++row)
   {
      // size if the columns of W must be = to length of p
      if (W[row].size() != P_COLS)
         throw std::invalid_argument("Array size mismatch in vectorMultiply");

      // compute the output for each row
      rowVal = 0.0;

      for (col = 0; col < W[row].size(); ++col)
         rowVal += W[row][col] * p[VECT_IDX];

      out.push_back(rowVal);
   }
} // vectorMultiply

/*=============================================================================
   vectorMultiply (could also be done as operator override)
      multiplies two vectors
=============================================================================*/
void vectorMultiply(const vector<double> &W,
                                    const vector<double> &p, double &out)
{
   // throw exception, note that if p.size > W.size the algorithm would appear
   // to work but the resulting output might be different than expected
   if (W.size() != p.size())
      throw std::invalid_argument("Array size mismatch in vectorMultiply");

   out = 0.0;

   for (unsigned int i = 0; i < W.size(); ++i)
      out += W[i] * p[i];
} // vectorMultiply

void Scale(vector<double> &vect, const double scale)
{
    unsigned int i;
    unsigned int numSamples = vect.size();

    for (i = 0; i < numSamples; ++i)
        vect[i] *= scale;
}

void Normalize(vector<double> &vect)
{
    double max = MaxAbsVal(vect);
    unsigned int numSamples = vect.size();

    if (max <= 0.0) max = 1.0;

    for (unsigned int i = 0; i < numSamples; ++i)
        vect[i] /= max;
}

double MaxAbsVal(const vector<double> &vect)
{
    double max = 0.0;
    unsigned int numSamples = vect.size();

    for (unsigned int i = 0; i < numSamples; ++i)
    {
        if (fabs(vect[i]) > max) max = fabs(vect[i]);
    }

    return max;
}
