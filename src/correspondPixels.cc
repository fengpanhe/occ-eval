
#include <string.h>
// #include <mex.h>
#include "Matrix.hh"
#include "csa.hh"
#include "match.hh"
#include "correspondPixels.h"
//#include <iostream>
//using namespace std;
static const double maxDistDefault = 0.0075;
static const double outlierCostDefault = 100;

void correspondPixels(double* match1, double* match2, double* cost_oc,
double* bmap1, double* bmap2, int h, int w, double maxDist = 0.0075, double outlierCost = 100)
{
    // do the computation
//    cout << bmap1[0] << ' ' << bmap1[1] << ' ' << bmap1[2] << endl;
    const int rows = h;
    const int cols = w;
    const double idiag = sqrt(rows * rows + cols * cols);
    const double oc = outlierCost * maxDist * idiag;
    Matrix m1, m2;
    const double cost = matchEdgeMaps(
        Matrix(rows, cols, bmap1), Matrix(rows, cols, bmap2),
        maxDist * idiag, oc,
        m1, m2);
    memcpy(match1, m1.data(), m1.numel() * sizeof(double));
    memcpy(match2, m2.data(), m2.numel() * sizeof(double));
    cost_oc[0] = cost;
    cost_oc[1] = oc;
}

// extern "C" {

// static const double maxDistDefault = 0.0075;
// static const double outlierCostDefault = 100;

// void
// mexFunction (
//     int nlhs, mxArray* plhs[],
//     int nrhs, const mxArray* prhs[])
// {
//     // check number of arguments
//     if (nlhs < 2) {
//         mexErrMsgTxt("Too few output arguments.");
//     }
//     if (nlhs > 4) {
//         mexErrMsgTxt("Too many output arguments.");
//     }
//     if (nrhs < 2) {
//         mexErrMsgTxt("Too few input arguments.");
//     }
//     if (nrhs > 4) {
//         mexErrMsgTxt("Too many input arguments.");
//     }

//     // get arguments
//     double* bmap1 = mxGetPr(prhs[0]);
//     double* bmap2 = mxGetPr(prhs[1]);
//     const double maxDist =
//         (nrhs>2) ? mxGetScalar(prhs[2]) : maxDistDefault;
//     const double outlierCost =
//         (nrhs>3) ? mxGetScalar(prhs[3]) : outlierCostDefault;

//     // check arguments
//     if (mxGetM(prhs[0]) != mxGetM(prhs[1])
//         || mxGetN(prhs[0]) != mxGetN(prhs[1])) {
//         mexErrMsgTxt("bmap1 and bmap2 must be the same size");
//     }
//     if (maxDist < 0) {
//         mexErrMsgTxt("maxDist must be >= 0");
//     }
//     if (outlierCost <= 1) {
//         mexErrMsgTxt("outlierCost must be >1");
//     }

//     // do the computation
//     const int rows = mxGetM(prhs[0]);
//     const int cols = mxGetN(prhs[0]);
//     const double idiag = sqrt( rows*rows + cols*cols );
//     const double oc = outlierCost*maxDist*idiag;
//     Matrix m1, m2;
//     const double cost = matchEdgeMaps(
//         Matrix(rows,cols,bmap1), Matrix(rows,cols,bmap2),
//         maxDist*idiag, oc,
//         m1, m2);

//     // set output arguments
//     plhs[0] = mxCreateDoubleMatrix(rows, cols, mxREAL);
//     plhs[1] = mxCreateDoubleMatrix(rows, cols, mxREAL);
//     double* match1 = mxGetPr(plhs[0]);
//     double* match2 = mxGetPr(plhs[1]);
//     memcpy(match1,m1.data(),m1.numel()*sizeof(double));
//     memcpy(match2,m2.data(),m2.numel()*sizeof(double));
//     if (nlhs > 2) { plhs[2] = mxCreateScalarDouble(cost); }
//     if (nlhs > 3) { plhs[3] = mxCreateScalarDouble(oc); }
// }

// }; // extern "C"
