# occ-eval
Evaluation code for object occlusion boundary detection

c++ 代码大部分参考 [pdollar/edges](https://github.com/pdollar/edges) 和 [BIDS/BSDS500](https://github.com/BIDS/BSDS500)

两部分代码都是适用于 matlab 的，修正了部分：

1. 传数组的内存顺序问题，matlab 的是列优先，python 是行优先。修改位置：
    + `Matrix.cc` 的 `double& operator() (int row, int col);` 和 `const double& operator() (int row, int col) const;`
    + `match.cc` 的　`323` 和 `328`　行
    

