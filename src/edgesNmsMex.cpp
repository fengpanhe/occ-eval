/*******************************************************************************
* Structured Edge Detection Toolbox      Version 3.01
* Code written by Piotr Dollar, 2014.
* Licensed under the MSR-LA Full Rights License [see license.txt]
*******************************************************************************/
// #include <mex.h>
#include <math.h>
#ifdef USEOMP
#include <omp.h>
#endif
 #include <iostream>
 using namespace std;

int h_ =  0, w_ = 0;

int getIndex(int x, int y)
{
    return y * w_ + x;
}

// return I[x,y] via bilinear interpolation
inline float interp(float* I, int h, int w, float x, float y)
{
    x = x < 0 ? 0 : (x > w - 1.001 ? w - 1.001 : x);
    y = y < 0 ? 0 : (y > h - 1.001 ? h - 1.001 : y);
    int x0 = int(x), y0 = int(y), x1 = x0 + 1, y1 = y0 + 1;
    float dx0 = x - x0, dy0 = y - y0, dx1 = 1 - dx0, dy1 = 1 - dy0;
    return I[getIndex(x0, y0)] * dx1 * dy1 + I[getIndex(x1, y0)] * dx0 * dy1 + I[getIndex(x0, y1)] * dx1 * dy0 + I[getIndex(x1, y1)] * dx0 * dy0;
}



extern "C" void edgesNms(float* E0, float* O, float* E, int h, int w, int r, int s, float m, int nThreads)
{
    // cout << "m:" << m << endl;
    h_ = h;
    w_ = w;
    cout << h << ' ' << w << '\n';
    cout << E0[0] << ' ' << E0[1] << ' ' << E0[2] << ' ' << E0[3] << '\n';
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++) {
            float e = E[getIndex(x, y)] = E0[getIndex(x, y)];
            if (!e)
                continue;
            e *= m;
            float coso = cos(O[getIndex(x, y)]), sino = sin(O[getIndex(x, y)]);
            for (int d = -r; d <= r; d++)
                if (d) {
                    float e0 = interp(E0, h, w, x + d * coso, y + d * sino);
                    if (e < e0) {
                        E[getIndex(x, y)] = 0;
                        break;
                    }
                }
        }
    // suppress noisy edge estimates near boundaries
    s = s > w / 2 ? w / 2 : s;
    s = s > h / 2 ? h / 2 : s;
    for (int x = 0; x < s; x++)
        for (int y = 0; y < h; y++) {
            E[getIndex(x, y)] *= x / float(s);
            E[getIndex((w - 1 - x), y)] *= x / float(s);
        }
    for (int x = 0; x < w; x++)
        for (int y = 0; y < s; y++) {
            E[getIndex(x, y)] *= y / float(s);
            E[getIndex(x, (h - 1 - y))] *= y / float(s);
        }
}
