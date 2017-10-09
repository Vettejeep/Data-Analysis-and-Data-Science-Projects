//******************************************************************************
//  FILE NAME:      adaline.h
//  DESCRIPTION:    class to implement a widrow-hoff style adaptive linear
//                  combiner with LMS back propagation
//******************************************************************************
#ifndef ADALINE_H
#define ADALINE_H

#include "globalfunctions.h"

#include<vector>
#include<stack>

using namespace std;

// notes:
// 1. designed so that the bias is part of the weight matrix, append a 1 to the
// input vector

class Adaline
{
public:
    Adaline() { mu = .02; mTransferFunc = logSig; }
    virtual ~Adaline() {}

    // inputs are kept const because they may be used by other nets
    void Run(const vector<double> &input, vector<double> &output);

    // set the transfer function, return false if null
    void SetXferFunc(void (*xferFunc)(const vector<double> &input,
                                                vector<double> &output))
                                                { mTransferFunc = xferFunc; }

    // set and get the weight vector
    void InitWeightsRandom(unsigned int rows,
                           unsigned int cols,
                           const double maxVal = 0.02);

    bool SetWeights(const vector<vector<double> > &weights);
    const vector<vector<double> >& GetWeights() const { return mWeights; }
    void GetNumWeights(unsigned int &rows, unsigned int &cols) const ;

    // set adaptation rate
    void SetMu(double u) { if (u <= 1.0 && u > 0.0) mu = u; else mu = .02; }
    double GetMu() { return mu; }

    // learning rules
    // LMS requires differentiable transfer function
    // adapts to, and passes back propagation information through: 'error'
    // if adapt is false, it is just used to backpropagate error
    bool AdaptLMS(vector<double> &error, bool adapt = true);

    void PushInputSet(vector<double> d) { mInputSet.push(d); }
    bool PopInputSet(vector<double> &v);
    void ClearInputSet();
    unsigned int GetInputSetSize() { return mInputSet.size(); }

    void PushDerivativeSet(vector<double> d) { mDerivativeSet.push(d); }
    bool PopDerivativeSet(vector<double> &v);
    void ClearDerivitiveSet();
    unsigned int GetDerivativeSetSize() { return mDerivativeSet.size(); }

protected:
    vector<vector<double> > mWeights;  // bias is built into mWeights
    double mu; // 2mu from the derivation, but can save the multiply

    // pointer to the transfer function for this layer
    void (*mTransferFunc)(const vector<double> &input, vector<double> &output);

    void CalcDerivative(const vector<double> &in);

    stack<vector<double> > mInputSet;
    stack<vector<double> > mDerivativeSet;
};

#endif // ADALINE_H
