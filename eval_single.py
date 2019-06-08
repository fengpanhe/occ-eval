import ctypes
import os
import numpy as np
from PIL import Image
from scipy import signal
import h5py

# from numpy.ctypeslib import ndpointer


def _get_mod(so_file):
    path = os.path.join(*(os.path.split(__file__)[:-1] + (so_file, )))
    mod = ctypes.cdll.LoadLibrary(path)
    return mod


_mod = _get_mod('./build/lib/libocc_eval.so')
_c_float_array2d = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2)
_c_double_array2d = np.ctypeslib.ndpointer(dtype=np.float, ndim=2)
_c_double_array1d = np.ctypeslib.ndpointer(dtype=np.float, ndim=1)

_edges_nms = _mod.edgesNms
_edges_nms.argtypes = [
    _c_float_array2d, _c_float_array2d, _c_float_array2d, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int
]
_edges_nms.restype = None


def c_edges_nms(edge_pr, ori, r, s, m, n_threads):
    if edge_pr.dtype != np.float32:
        edge_pr = edge_pr.astype(np.float32)
    if ori.dtype != np.float32:
        ori = ori.astype(np.float32)
    h, w = edge_pr.shape
    thin_edge = np.zeros_like(edge_pr)
    _edges_nms(edge_pr, ori, thin_edge, h, w, r, s, m, n_threads)
    return thin_edge


_correspond_pixels = _mod.correspondPixels
_correspond_pixels.argtypes = [
    _c_double_array2d, _c_double_array2d, _c_double_array1d, _c_double_array2d,
    _c_double_array2d, ctypes.c_int, ctypes.c_int, ctypes.c_double,
    ctypes.c_double
]
_correspond_pixels.restype = None


def c_correspond_pixels(map1, map2, max_dist=0.0075, outlier_cost=100):
    if map1.dtype != np.float:
        map1 = map1.astype(np.float)
    if map2.dtype != np.float:
        map2 = map2.astype(np.float)
    if map1.shape != map2.shape:
        print('c_correspond_pixels: map1 和 map2 大小不匹配。')
        exit()
    h, w = map1.shape
    match1 = np.zeros_like(map1)
    match2 = np.zeros_like(map2)
    cost_oc = np.zeros(2, dtype=np.float)
    _correspond_pixels(match1, match2, cost_oc, map1, map2, h, w, max_dist,
                       outlier_cost)
    return match1, match2, cost_oc[0], cost_oc[1]


_occ_eval = _mod.occEval
_occ_eval.argtypes = [
    _c_double_array2d, ctypes.c_int, ctypes.c_int, _c_double_array2d,
    _c_double_array2d, _c_double_array2d, _c_double_array2d, ctypes.c_int,
    ctypes.c_int, _c_double_array1d, ctypes.c_int, ctypes.c_double,
    ctypes.c_double
]
_occ_eval.restype = None


def c_occ_eval(edge_pb, edge_gt, ori_pb, ori_gt, opt):
    if edge_pb.dtype != np.float:
        edge_pb = edge_pb.astype(np.float)
    if edge_gt.dtype != np.float:
        edge_gt = edge_gt.astype(np.float)
    if ori_pb.dtype != np.float:
        ori_pb = ori_pb.astype(np.float)
    if ori_gt.dtype != np.float:
        ori_gt = ori_gt.astype(np.float)
    h, w = edge_pb.shape

    thresholds_num = opt['nthresh']
    thresholds = np.linspace(0, 1, thresholds_num + 2)[1:-1]
    res = np.zeros((thresholds_num, 9))
    res[:, 0] = thresholds

    ori_diff = opt['ori_diff']
    thread_num = opt['nthread']
    max_dist = opt['maxDist']
    _occ_eval(res, thresholds_num, 9, edge_pb, edge_gt, ori_pb, ori_gt, h, w,
              ori_diff, thread_num, max_dist, 100)
    thresholds = res[:, 0]
    res = res[:, 1:]
    return thresholds, res


def triangle_filter(n):
    f = np.zeros((1 + 2 * n))
    for i in range(n):
        f[i] = i + 1
        f[-i - 1] = i + 1
    f[n] = n + 1
    f = f / np.sum(f)
    ff = np.dot(np.transpose([f]), [f])
    return ff


def conv_tri(I, r):
    ff = triangle_filter(r)
    J = np.pad(I, (r, r), mode='symmetric')
    res = signal.convolve2d(J, ff, mode='valid')
    return res


def edges_nms(edge_pr):
    if edge_pr.dtype != np.float32:
        edge_pr = edge_pr.astype(np.float32)
    Oy, Ox = np.gradient(conv_tri(edge_pr, 4))
    _, Oxx = np.gradient(Ox)
    Oyy, Oxy = np.gradient(Oy)
    ori = np.mod(np.arctan(Oyy * np.sign(-Oxy) / (Oxx + 1e-5)), np.pi)
    thin_edge = c_edges_nms(edge_pr, ori, 1, 5, 1.01, 4)
    return thin_edge


def occ_eval_single(pb, gt, opt_in={}):
    opt = {
        'nthresh': 99,
        'maxDist': 0.0075,
        'nthread': 25,
        'ori_diff': np.array([np.pi / 2, np.pi * 3 / 2])
    }
    opt = {**opt, **opt_in}

    if pb.dtype != np.float:
        pb = pb.astype(np.float)
    if gt.dtype != np.float:
        gt = gt.astype(np.float)
    if pb.shape != gt.shape:
        print('c_correspond_pixels: map1 和 map2 大小不匹配。')
        exit()

    edge_pb = pb[0, :, :]
    ori_pb = pb[1, :, :]
    edge_gt = gt[0, :, :]
    ori_gt = gt[1, :, :]

    edge_pb = edges_nms(edge_pb)
    nozero = np.where(edge_pb != 0)
    edge_pb[nozero] = (edge_pb[nozero] - np.min(edge_pb[nozero])) / np.max(
        edge_pb[nozero])

    thresholds, res = c_occ_eval(edge_pb, edge_gt, ori_pb, ori_gt, opt)
    return thresholds, res


import matplotlib.pyplot as plt


def showimg(a):
    img = plt.imshow(a)
    img.set_cmap('gray')
    plt.show()


if __name__ == "__main__":
    # I = Image.open('./100007.png')
    # a = np.array(I, dtype=np.float) / 255
    # b = edges_nms(a)
    # showimg(b)
    # print(b)
    # print(type(b))
    f = h5py.File('./data/2008_001028.h5', 'r')
    edge_pb = f['edge']
    ori_pb = f['ori_map']
    edge_ori_gt = f['edge_ori_gt']
    c_occ_eval(
        np.stack((edge_pb, ori_pb)),
        np.stack((edge_ori_gt[0, :, :], edge_ori_gt[1, :, :])))
    f.close()
    pass