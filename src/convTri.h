#ifndef __CONVTRI_H__
#define __CONVTRI_H__
#include <iostream>
using namespace std;
void convTri(float *I, float *O, int h, int w, int d, int r, int s);
extern "C" void convTriangle(float *I, float *O, int h, int w, int d, int r, int s){
  cout << I[0] << ' ' << I[1] << ' ' <<I[481] << endl;
  cout << h << ' ' << w << ' ' <<d << ' ' << r << ' ' << s << endl;
  float* Ot = new float[h*w];
  convTri(I, Ot, h, w, d, r, s);
  cout << Ot[0] << ' ' << Ot[1] << endl;
  delete Ot;
}
#endif