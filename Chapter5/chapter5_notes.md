## 时序差分方法

在$Chapter3$中，我们提到过蒙特卡洛方法，当时我们给出的蒙特卡洛法迭代式为$$V(s_t) \leftarrow V(s_t) + \frac{1}{N(s)}[G_t - V(s_t)]$$

在《Reinforcement Learning: An Introduction》书中，给出了一个更普遍的蒙特卡洛法迭代式$$V(s_t) \leftarrow V(s_t) + \alpha[G_t - V(s_t)]$$

对于式子中的$\alpha$更一般的表达是$\alpha_n(a)$，当满足下面两个条件时$$\sum_{n=1}^{\infty} \alpha_n(a) = \infty \quad \text{和} \quad \sum_{n=1}^{\infty} \alpha_n^2(a) < \infty$$

利用Robbins-Monro定理，可以证明当$\alpha_n(a) = \frac{1}{N}$时迭代式收敛，并且收敛到价值函数的真值上

当我们给出的$\alpha$是常数，并且$0 < \alpha < 1$时，这种情况下并不满足第二个条件，这代表我们迭代式不会完全收敛

不过我们可以证明，递推式的结果是价值函数真值的无偏期望，通过递推表达式我们可以求出具体的计算式$$V^{(k)}(s) = (1-\alpha)^{k-1} G_t^{(1)} + \alpha \sum_{i=2}^{k} (1-\alpha)^{k-i} G_t^{(i)}$$

我们对$V^{(k)}(s)$求期望可以得到$$E[V^{(k)}(s)] = V^{\pi}(s) \left[ (1-\alpha)^{k-1} + \alpha \sum_{i=2}^{k} (1-\alpha)^{k-i} \right] = V^{\pi}(s)$$

上式证明了，递推式的结果是价值函数真值的无偏期望，同时我们不妨记$d^{2} = D(G_t)$那么有

$$
\begin{aligned}
D(V^{(k)}(s)) &= d^2 \left[ \alpha^2 \sum_{i=2}^{k} (1-\alpha)^{2k-2i} + (1-\alpha)^{2k-2} \right] \\
&= d^2 \left[ \frac{\alpha}{2-\alpha} + \frac{2(1-\alpha)^{2k-1}}{2-\alpha} \right]
\end{aligned}
$$

当 $k \rightarrow \infty$ 时有 $$D(V^{(k)}(s)) \rightarrow \frac{\alpha d^2}{2-\alpha}$$

我们发现$D(V^{(k)}(s))$不收敛到$0$，下面证明 $V^{(k)}(s)$ 不依概率收敛到 $V^{\pi}(s)$

不妨假设$V^{(k)}(s) \xrightarrow{P} V^{\pi}(s)$成立，即$V^{(k)}(s) \xrightarrow{P} C$ (其中$C$为常数)

那么有$V^{(k)}(s) \xrightarrow{d} C$，这意味着$k \rightarrow \infty$时，$V^{(k)}(s) \sim \delta(C)$

那么可以推出来$$\lim_{k \rightarrow \infty} D(V^{(k)}(s)) = 0$$

这与我们上面的结果矛盾！

所以$V^{(k)}(s)$ 不依概率收敛到 $V^{\pi}(s)$

所以根据上面的证明，我们可以最终下结论，对于$\alpha$是常数的蒙特卡洛法迭代式，最终不会收敛到一个固定的常数值，而会最终随着迭代在这个方差范围内波动

