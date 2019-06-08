import numpy as np
from .eval_single import occ_eval_single


def f1measure(p_count, p_sum, r_count, r_sum):
    p = p_count / (p_sum + float(p_sum == 0))
    r = r_count / (r_sum + float(r_sum == 0))
    f_measure = 2 * p * r / (p + r + float((p + r) == 0))
    return f_measure


def max_f1measure(thresholds, p_count, p_sum, r_count, r_sum):
    # 先进行插值，每两个阈值中间插入 9 个值
    n = p_count.shape[0]
    new_n = (n - 1) * 9 + n
    new_x = np.arange(new_n)
    xp = np.arange(0, new_n, 10)
    thresholds = np.interp(new_x, xp, thresholds)
    p_count = np.interp(new_x, xp, p_count)
    p_sum = np.interp(new_x, xp, p_sum)
    r_count = np.interp(new_x, xp, r_count)
    r_sum = np.interp(new_x, xp, r_sum)

    p = p_count / (p_sum + (p_sum == 0).astype(float))
    r = r_count / (r_sum + (r_sum == 0).astype(float))
    f_measure = 2 * p * r / (p + r + ((p + r) == 0).astype(float))
    max_f_index = np.argmax(f_measure)
    return thresholds[max_f_index], f_measure[max_f_index], p_count[
        max_f_index], p_sum[max_f_index], r_count[max_f_index], r_sum[
            max_f_index]


def pr_area(p_count, p_sum, r_count, r_sum):
    p = p_count / (p_sum + (p_sum == 0).astype(float))
    r = r_count / (r_sum + (r_sum == 0).astype(float))
    inter_p = np.interp(np.linspace(0, 1, 101), r, p)
    area = np.sum(inter_p) / 100
    return area


class EvalOcc(object):
    def __init__(self, max_dist=0.0075, num_thresh=99):
        self.max_dist = max_dist
        self.num_thresh = num_thresh

        self.thresholds = np.linspace(0, 1, self.num_thresh + 2)[1:-1]

        self.edge_precision_count = np.zeros(self.num_thresh)
        self.edge_precision_sum = np.zeros(self.num_thresh)
        self.edge_recall_count = np.zeros(self.num_thresh)
        self.edge_recall_sum = np.zeros(self.num_thresh)
        self.ori_precision_count = np.zeros(self.num_thresh)
        self.ori_recall_count = np.zeros(self.num_thresh)

        self.edge_best_precision_count = 0.0
        self.edge_best_precision_sum = 0.0
        self.edge_best_recall_count = 0.0
        self.edge_best_recall_sum = 0.0
        self.ori_best_precision_count = 0.0
        self.ori_best_precision_sum = 0.0
        self.ori_best_recall_count = 0.0
        self.ori_best_recall_sum = 0.0

    def reset(self):
        self.edge_precision_count = np.zeros(self.num_thresh)
        self.edge_precision_sum = np.zeros(self.num_thresh)
        self.edge_recall_count = np.zeros(self.num_thresh)
        self.edge_recall_sum = np.zeros(self.num_thresh)
        self.ori_precision_count = np.zeros(self.num_thresh)
        self.ori_recall_count = np.zeros(self.num_thresh)

        self.edge_best_precision_count = 0.0
        self.edge_best_precision_sum = 0.0
        self.edge_best_recall_count = 0.0
        self.edge_best_recall_sum = 0.0
        self.ori_best_precision_count = 0.0
        self.ori_best_precision_sum = 0.0
        self.ori_best_recall_count = 0.0
        self.ori_best_recall_sum = 0.0

    def add_single(self, pb, gt):
        thresholds, res = occ_eval_single(pb.cpu().numpy(), gt.cpu().numpy())
        edge_p_count = res[:, 0]
        edge_p_sum = res[:, 1]
        edge_r_count = res[:, 2]
        edge_r_sum = res[:, 3]
        ori_p_count = res[:, 4]
        ori_p_sum = res[:, 5]
        ori_r_count = res[:, 6]
        ori_r_sum = res[:, 7]

        self.edge_precision_count += edge_p_count
        self.edge_precision_sum += edge_p_sum
        self.edge_recall_count += edge_r_count
        self.edge_recall_sum += edge_r_sum
        self.ori_precision_count += ori_p_count
        self.ori_recall_count += ori_r_count

        _, _, best_p_c, best_p_s, best_r_c, best_r_s = max_f1measure(
            thresholds, edge_p_count, edge_p_sum, edge_r_count, edge_r_sum)
        self.edge_best_precision_count += best_p_c
        self.edge_best_precision_sum += best_p_s
        self.edge_best_recall_count += best_r_c
        self.edge_best_recall_sum += best_r_s

        _, _, best_p_c, best_p_s, best_r_c, best_r_s = max_f1measure(
            thresholds, ori_p_count, ori_p_sum, ori_r_count, ori_r_sum)
        self.ori_best_precision_count += best_p_c
        self.ori_best_precision_sum += best_p_s
        self.ori_best_recall_count += best_r_c
        self.ori_best_recall_sum += best_r_s

    def get_eval_res(self):
        _, edge_ods, _, _, _, _ = max_f1measure(
            self.thresholds, self.edge_precision_count,
            self.edge_precision_sum, self.edge_recall_count,
            self.edge_recall_sum)
        edge_ois = f1measure(
            self.edge_best_precision_count, self.edge_best_precision_sum,
            self.edge_best_recall_count, self.edge_best_recall_sum)
        edge_ap = pr_area(self.edge_precision_count, self.edge_precision_sum,
                          self.edge_recall_count, self.edge_recall_sum)

        _, ori_ods, _, _, _, _ = max_f1measure(
            self.thresholds, self.ori_precision_count, self.edge_precision_sum,
            self.ori_recall_count, self.edge_recall_sum)
        ori_ois = f1measure(
            self.ori_best_precision_count, self.ori_best_precision_sum,
            self.ori_best_recall_count, self.ori_best_recall_sum)
        ori_ap = pr_area(self.ori_precision_count, self.edge_precision_sum,
                         self.ori_recall_count, self.edge_recall_sum)
        return edge_ods, edge_ois, edge_ap, ori_ods, ori_ois, ori_ap
