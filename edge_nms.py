import ctypes
import os
import numpy as np
from PIL import Image
from scipy import signal
# from numpy.ctypeslib import ndpointer


def _get_mod(so_file):
    path = os.path.join(*(os.path.split(__file__)[:-1] + (so_file, )))
    mod = ctypes.cdll.LoadLibrary(path)
    return mod


_float_c = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2)

# _conv_tri = _get_mod('./build/lib/libconvTriangle.so').convTriangle
# _conv_tri.argtypes = [
#     _float_c, _float_c, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
#     ctypes.c_int
# ]
# _conv_tri.restype = None

# _gradient2 = _get_mod('./build/lib/libgradient2.so').gradient2
# _gradient2.argtypes = [
#     _float_c, _float_c, _float_c, ctypes.c_int, ctypes.c_int, ctypes.c_int
# ]
# _gradient2.restype = None

_edges_nms = _get_mod('./build/lib/libedges_nms.so').edgesNms
_edges_nms.argtypes = [
    _float_c, _float_c, _float_c, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_float, ctypes.c_int
]
_edges_nms.restype = None


def triangle_filter(n):
    f = np.zeros((1+2*n))
    for i in range(n):
        f[i] = i+1
        f[-i-1] = i+1
    f[n] = n + 1
    f = f / np.sum(f)
    ff = np.dot(np.transpose([f]),[f])
    return ff


def conv_tri(I, r):
    # if I.dtype != np.float32:
    #     I = I.astype(np.float32)
    # s = 1
    # d = 1
    # if len(I.shape) == 3:
    #     d, h, w = I.shape
    # elif len(I.shape) == 2:
    #     h, w = I.shape
    # else:
    #     print('conv_tri: the dimensions of I is wrong.', )
    # output = np.zeros_like(I)
    # _conv_tri(I, output, h, w, d, r, s)
    # return output
    ff = triangle_filter(r)
    J = np.pad(I, (r, r), mode='symmetric')
    res = signal.convolve2d(J, ff, mode='valid')
    return res


def gradient2(I):
    if I.dtype != np.float32:
        I = I.astype(np.float32)
    d = 1
    if len(I.shape) == 3:
        d, h, w = I.shape
    elif len(I.shape) == 2:
        h, w = I.shape
    else:
        print('conv_tri: the dimensions of I is wrong.', )
    gradient_x = np.zeros_like(I)
    gradient_y = np.zeros_like(I)
    _gradient2(I, gradient_x, gradient_y, h, w, d)
    return gradient_x, gradient_y


def c_edges_nms(edge_pr, ori, r, s, m, n_threads):
    if edge_pr.dtype != np.float32:
        edge_pr = edge_pr.astype(np.float32)
    if ori.dtype != np.float32:
        ori = ori.astype(np.float32)
    h, w = edge_pr.shape
    thin_edge = np.zeros_like(edge_pr)
    _edges_nms(edge_pr, ori, thin_edge, h, w, r, s, m, n_threads)
    return thin_edge


def edges_nms(edge_pr):
    if edge_pr.dtype != np.float32:
        edge_pr = edge_pr.astype(np.float32)
    Oy, Ox = np.gradient(conv_tri(edge_pr, 4))
    _, Oxx = np.gradient(Ox)
    Oyy, Oxy = np.gradient(Oy)
    ori = np.mod(np.arctan(Oyy * np.sign(-Oxy) / (Oxx + 1e-5)), np.pi)
    thin_edge = c_edges_nms(edge_pr, ori, 1, 5, 1.01, 4)
    return thin_edge


import matplotlib.pyplot as plt

def showimg(a):
    img = plt.imshow(a)
    img.set_cmap('gray')
    plt.show()


if __name__ == "__main__":
    I = Image.open('./100007.png')
    a = np.array(I, dtype=np.float) / 255
    # a = np.array(list(range(300 * 300)), dtype=np.float)
    # a = a.reshape((300, 300))
    # print(a)
    # b = conv_tri(a, 4)
    b = edges_nms(a)
    showimg(b)
    print(b)
    # print(type(b))
    pass