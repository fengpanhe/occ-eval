#ifndef __CORRESPONDPIXELS_hh__
#define __CORRESPONDPIXELS_hh__

extern "C" void correspondPixels(double* match1, double* match2, double* cost_oc,
double* bmap1, double* bmap2, int h, int w, double maxDist, double outlierCost);
#endif