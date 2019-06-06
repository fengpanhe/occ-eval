#ifndef OCCEVAL_H
#define OCCEVAL_H

#include "Matrix.hh"
#include "csa.hh"
#include "match.hh"
#include "ThreadPool.h"
// #include "MatchPixelNumber.h"
// class MatchPixelNumber;
// res 应为 res_rows*9 大小的数组，第一列为 edge 概率的阈值，后 8 列为结果值。
extern "C" void occEval(double* res, int res_rows, int res_cols, 
double* edge_pb, double* edge_gt, double* ori_pb, double* ori_gt, int h, int w, 
double* ori_diff, int thread_num = 8, double maxDist = 0.0075, double outlierCost = 100);


class MatchPixelNumber : public ThreadClass {
public:
    static int rows;
    static int cols;
    static double dist;
    static double oc;
    static double ori_angle_range[2];
    static double* edge_pb;
    static double* edge_gt;
    static double* ori_pb;
    static double* ori_gt;
    static int edge_r_sum;
    static int ori_r_sum;
    bool complete;
    MatchPixelNumber(double threshold, double* res);
    ~MatchPixelNumber();
    void run() override;

private:
    double threshold_;
    double* res_;
};
#endif