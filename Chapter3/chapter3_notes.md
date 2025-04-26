## 马尔可夫奖励过程

对于马尔可夫奖励过程中的贝尔曼方程$$\mathcal{V} = \mathcal{R} + \gamma P \mathcal{V}$$

其解析解$$\mathcal{V} = (I - \gamma P)^{-1} \mathcal{R}$$

中的$I - \gamma P$矩阵，如果$0 \le \gamma < 1$，那么这个矩阵一定可逆

笔者查找到的证明方式大约有两种

- 基于谱半径的证明
  - [维基百科 - Stochastic Matrix](https://en.wikipedia.org/wiki/Stochastic_matrix#Spectral_radius)。
- 基于冯诺依曼级数的证明
  - [维基百科 - Neumann Series](https://en.wikipedia.org/wiki/Neumann_series#Matrix_case)

## 占用度量

占用度量在本书中的定义是$$\rho^\pi(s, a) = (1 - \gamma) \sum_{t=0}^\infty \gamma^t P_t^\pi(s) \pi(a|s)$$

在课程中对于占用度量的定义是$$\rho^\pi(s, a) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t \mathbb{I}(S_t = s, A_t = a) \mid \pi \right] = \sum_{t=0}^\infty \gamma^t P(S_t = s, A_t = a \mid s_0, \pi)$$

这两种定义唯一的区别便是第一个用$1 - \gamma$来进行归一化，而第二个定义并没有，这两种都已都是有效的

对于归一化的占用度量，会倾向于把它当作一个概率分布来使用，对于未归一化的占用度量，通常我们使用它来进行直接计算状态价值函数$$V^\pi(s_0) = \sum_{s,a} \sum_{t=0}^T \gamma^t P(S_t = s, A_t = a \mid s_0, \pi) r(s, a) = \sum_{s, a} \rho^\pi(s, a) r(s, a)$$

不过我们只可以直接计算状态价值函数，而无法直接使用占用度量计算动作价值函数，不过我们可以通过直接计算出状态价值函数后，使用状态价值函数间接计算动作价值函数$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s')$$


