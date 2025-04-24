## 马尔可夫奖励过程

对于马尔可夫奖励过程中的贝尔曼方程$$\mathcal{V} = \mathcal{R} + \gamma P \mathcal{V}$$

其解析解$$\mathcal{V} = (I - \gamma P)^{-1} \mathcal{R}$$

中的$I - \gamma P$矩阵，如果$0 \le \gamma < 1$，那么这个矩阵一定可逆

笔者查找到的证明方式大约有两种

- 基于谱半径的证明
  - [维基百科 - Stochastic Matrix](https://en.wikipedia.org/wiki/Stochastic_matrix#Spectral_radius)。
- 基于冯诺依曼级数的证明
  - [维基百科 - Neumann Series](https://en.wikipedia.org/wiki/Neumann_series#Matrix_case)