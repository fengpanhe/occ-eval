#include "OccEval.h"
#include "iostream"
#define EPS 1e-6
#define PI acos(-1)
using namespace std;
void occEval(double* res, int res_rows, int res_cols,
    double* edge_pb, double* edge_gt, double* ori_pb, double* ori_gt, int h, int w,
    double* ori_diff, int thread_num, double maxDist, double outlierCost)
{
    const double idiag = sqrt(h * w + h * w);
    MatchPixelNumber::rows = h;
    MatchPixelNumber::cols = w;
    MatchPixelNumber::dist = maxDist * idiag;
    MatchPixelNumber::oc = outlierCost * maxDist * idiag;
    MatchPixelNumber::ori_angle_range[0] = ori_diff[0];
    MatchPixelNumber::ori_angle_range[1] = ori_diff[1];
    MatchPixelNumber::edge_pb = edge_pb;
    MatchPixelNumber::edge_gt = edge_gt;
    MatchPixelNumber::ori_pb = ori_pb;
    MatchPixelNumber::ori_gt = ori_gt;
    int edge_r_sum = 0;
    for (int i = 0; i < h * w; i++) {
        if (edge_gt[i] > 0.5)
            edge_r_sum++;
    }
    MatchPixelNumber::edge_r_sum = edge_r_sum;
    MatchPixelNumber::ori_r_sum = edge_r_sum;
    ThreadPool* pool = nullptr;
    try {
        pool = new ThreadPool(thread_num);
    } catch (...) {
        return;
    }
    ThreadClass* mpn[100];
    for (int i = 0; i < res_rows; i++) {
        int offset = i * res_cols;
        mpn[i] = new MatchPixelNumber(res[offset], res + offset);
        pool->append(mpn[i]);
    }

    int complete_count = 0;
    MatchPixelNumber* tmp;
    while (complete_count < res_rows) {
        complete_count = 0;
        for (int i = 0; i < res_rows; i++) {
            tmp = (MatchPixelNumber*)mpn[i];
            if (tmp->complete)
                complete_count++;
        }
    }

    for (int i = 0; i < res_rows; i++) {
        delete mpn[i];
    }
    // delete mpn;
    delete pool;
}

int MatchPixelNumber::rows = 0;
int MatchPixelNumber::cols = 0;
double MatchPixelNumber::dist = 0.0;
double MatchPixelNumber::oc = 0.0;
double MatchPixelNumber::ori_angle_range[2];
double* MatchPixelNumber::edge_pb = nullptr;
double* MatchPixelNumber::edge_gt = nullptr;
double* MatchPixelNumber::ori_pb = nullptr;
double* MatchPixelNumber::ori_gt = nullptr;
int MatchPixelNumber::ori_r_sum = 0;
int MatchPixelNumber::edge_r_sum = 0;

MatchPixelNumber::MatchPixelNumber(double threshold, double* res)
{
    threshold_ = threshold;
    res_ = res;
    complete = false;
}

MatchPixelNumber::~MatchPixelNumber()
{
}

void MatchPixelNumber::run()
{
    int map_size = rows * cols;
    int edge_p_sum = 0;
    double* edge_map = new double[rows * cols];
    for (int i = 0; i < map_size; i++) {
        if ((edge_pb[i] - EPS) >= threshold_) {
            edge_p_sum++;
            edge_map[i] = 1;
        } else
            edge_map[i] = 0;
    }
    Matrix m1, m2;
    double cost = matchEdgeMaps(
        Matrix(rows, cols, edge_map), Matrix(rows, cols, edge_gt),
        dist, oc,
        m1, m2);
    int edge_p_count = 0;

    int edge_r_count = 0;
    int ori_p_count = 0;
    int ori_r_count = 0;
    for (int i = 0; i < map_size; i++) {
        if (m2(i) > 0.5)
            edge_r_count++;
        if (m1(i) > 0.5) {
            edge_p_count++;
            double diff = fabs(ori_pb[i] - ori_gt[int(m1(i))]);
            diff = fmod(diff, 2 * PI);
            if ((diff + EPS) <= ori_angle_range[0] || (diff - EPS)>= ori_angle_range[1]) {
                ori_p_count++;
                ori_r_count++;
            }
        }
    }
    int ori_p_sum = edge_p_sum;
    res_[0] = (double)threshold_;
    res_[1] = (double)edge_p_count;
    res_[2] = (double)edge_p_sum;
    res_[3] = (double)edge_r_count;
    res_[4] = (double)edge_r_sum;
    res_[5] = (double)ori_p_count;
    res_[6] = (double)ori_p_sum;
    res_[7] = (double)ori_r_count;
    res_[8] = (double)ori_r_sum;
    if (edge_map != nullptr) {
        delete edge_map;
    }
    complete = true;
}