//******************************************************************************
//  FILE NAME:      adaline.cpp
//  DESCRIPTION:    class to implement a widrow-hoff style adaptive linear
//                  combiner with LMS back propagation
//******************************************************************************
#include "globalfunctions.h"
#include <cstdlib>
#include <exception>

#include "Adaline.h"

//******************************************************************************
//  CLASS: Adaline
//******************************************************************************

//******************************************************************************
//  FUNCTION:	  Adaline::Run
//  DESCRIPTION:  compute output based on input and current system state,
//                i.e., the current training state of the weight matrix
//******************************************************************************
void Adaline::Run(const vector<double> &input, vector<double> &output)
{
    // note: bad array sizes will create an exception in vectorMultiply
    PushInputSet(input);

    vector<double> Wp;

    vectorMultiply(mWeights, input, Wp); // includes bias
    (*mTransferFunc)(Wp, output);
    CalcDerivative(Wp);
}

//******************************************************************************
//  FUNCTION:	  Adaline::AdaptLMS
//  DESCRIPTION:  adapt the weights based on back-propagation error,
//                passes out an equivalent error for previous layers
//******************************************************************************
bool Adaline::AdaptLMS(vector<double> &error, bool adapt)
{
    if (error.size() != mWeights.size() ||
            mDerivativeSet.empty() || mInputSet.empty())
        return false;

    vector<double> errOut;
    vector<double> derivative;
    vector<double> input;

    // stack stores errors, mostly useful for networks where error needs
    // to propagate back through many layers such as those where error
    // is not available at the end of each step
    if (!PopDerivativeSet(derivative)) return false;
    if (!PopInputSet(input)) return false;

    unsigned int i, j;

    for (j = 0; j < mWeights.size(); ++j)
    {
        error[j] = derivative[j] * error[j];

        for (i = 0; i < mWeights[j].size(); ++i)
        {
            // adapt: otherwise this function is used to back propagate
            // error through an emulator which is already trained
            if (adapt)
                mWeights[j][i] = mWeights[j][i] + (mu * error[j] * input[i]);

            if (j != 0)
                errOut[i] += mWeights[j][i] * error[j];
            else
                errOut.push_back(mWeights[j][i] * error[j]);
        }
    }

    error.clear();
    error.assign(errOut.begin(), errOut.end());

    return true;
}

//******************************************************************************
//  FUNCTION:	  Adaline::CalcDerivative
//  DESCRIPTION:  get the transfer function derivative
//******************************************************************************
// this will only work for a differentiable transfer function -
// add check for this? this is a generic derivative estimator
void Adaline::CalcDerivative(const vector<double> &in)
{
    vector<double> loIn, hiIn, loOut, hiOut;
    vector<double> derivative;

    for (unsigned int i = 0; i < in.size(); ++i)
    {
        loIn.push_back(in[i] - .0005);
        hiIn.push_back(in[i] + .0005);
    }

    (*mTransferFunc)(loIn, loOut);
    (*mTransferFunc)(hiIn, hiOut);

    for (unsigned int i = 0; i < in.size(); ++i)
        derivative.push_back((hiOut[i] - loOut[i]) * 1000.0);

    PushDerivativeSet(derivative);
}

//******************************************************************************
//  FUNCTION:	  Adaline::GetNumWeights
//  DESCRIPTION:  get the row and column sizes of the weight matrices
//******************************************************************************
void Adaline::GetNumWeights(unsigned int &rows, unsigned int &cols) const
{
    // ensure that rows are same length in setWeights
    rows = cols = 0;
    if (mWeights.size() > 0)
    {
        rows = mWeights.size();
        if (mWeights[0].size() > 0)
            cols = mWeights[0].size();
    }
}

//******************************************************************************
//  FUNCTION:	  Adaline::InitWeightsRandom
//  DESCRIPTION:  initialize weights to small random values to start training
//******************************************************************************
void Adaline::InitWeightsRandom(unsigned int rows,
                                unsigned int cols,
                                const double maxVal)
{
    double randVal;
    double randFact = maxVal * 2.0 / (double)RAND_MAX;

    mWeights.clear();
    if (rows == 0) rows = 1;
    if (cols == 0) cols = 1;

    for (unsigned int j = 0; j < rows; ++j)
    {
        vector<double> v;

        for (unsigned int i = 0; i < cols; ++i)
        {
            randVal = (rand() * randFact) - maxVal;
            v.push_back(randVal);
        }

        mWeights.push_back(v);
    }
}


//******************************************************************************
//  FUNCTION:	  Adaline::SetWeights
//  DESCRIPTION:  set the weights to a known state, such as from a file
//******************************************************************************
bool Adaline::SetWeights(const vector<vector<double> > &weights)
{
    mWeights.clear();
    unsigned int i;

    for (i = 0; i < weights.size(); ++i)
    {
        vector<double> v;
        v.assign(weights[i].begin(), weights[i].end());
        mWeights.push_back(v);
    }

    // return true if sizes seem OK - could check each row
    return (mWeights.size() > 0 && mWeights[0].size() > 0);
}

//******************************************************************************
//  FUNCTION:	  Adaline::PopDerivativeSet
//  DESCRIPTION:  pop one item off the derivative stack for training
//                these are stored items, so that a single object may be used
//                on multi-step runs where training error is unavailable until
//                the end
//******************************************************************************
bool Adaline::PopDerivativeSet(vector<double> &v)
{
    v.clear();
    vector<double> topElm;

    if (!mDerivativeSet.empty())
    {
        topElm = mDerivativeSet.top();
        v.assign(topElm.begin(), topElm.end());
        mDerivativeSet.pop();
        return true;
    }

    else
        return false;
}

//******************************************************************************
//  FUNCTION:	  Adaline::PopDerivativeSet
//  DESCRIPTION:  clear the derivative stack, reset for new training run
//  NOTE:         generally the stack should be cleared by back propagation,
//                however, this ensures it is empty
//******************************************************************************
void Adaline::ClearDerivitiveSet()
{
    while (!mDerivativeSet.empty())
        mDerivativeSet.pop();
}

//******************************************************************************
//  FUNCTION:	  Adaline::PopDerivativeSet
//  DESCRIPTION:  pop one item off the inputs stack for training,
//                these are stored inputs, so that a single object may be used
//                on multi-step runs where training error is unavailable until
//                the end
//******************************************************************************
bool Adaline::PopInputSet(vector<double> &v)
{
    v.clear();
    vector<double> topElm;

    if (!mInputSet.empty())
    {
        topElm = mInputSet.top();
        v.assign(topElm.begin(), topElm.end());
        mInputSet.pop();
        return true;
    }

    else
        return false;
}

//******************************************************************************
//  FUNCTION:	  Adaline::PopDerivativeSet
//  DESCRIPTION:  clear the derivative stack, reset for new training run
//  NOTE:         generally the stack should be cleared by back propagation,
//                however, this ensures it is empty
//******************************************************************************
void Adaline::ClearInputSet()
{
    while (!mInputSet.empty())
        mInputSet.pop();
}
