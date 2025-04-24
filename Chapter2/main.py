from typing import Callable
from bernoulli_bandit import BernoulliBandit
from epsilon_greedy import EpsilonGreedy
from decaying_epsilon_greedy import DecayingEpsilonGreedy
from ucb import UCB
from thompson_sampling import ThompsonSampling
from solver import Solver
import matplotlib.pyplot as plt
import numpy as np

def plot_result(solvers: list[Solver], names: list[str]) -> None:
    for index, solver in enumerate(solvers):
        x_values = range(len(solver.r_list))
        plt.plot(x_values, solver.r_list, label=names[index])
        plt.title(f"{solver.bandit.n} arm bandit")

    plt.xlabel("time")
    plt.ylabel("regret")
    plt.legend()
    plt.show()

def epsilon_greedy_run(epsilons: list[float] | None = None, t: int = 5000, n: int = 10) -> None:
    solvers = list()
    names = list()

    bernoulli_bandit_n = BernoulliBandit(n)

    if not epsilons:
        epsilon_greedy = EpsilonGreedy(bernoulli_bandit_n, 0.01)
        epsilon_greedy.run(t)
        solvers.append(epsilon_greedy)
        names.append("epsilon_greedy")
    else:
        for epsilon in epsilons:
            epsilon_greedy = EpsilonGreedy(bernoulli_bandit_n, epsilon)
            epsilon_greedy.run(t)
            solvers.append(epsilon_greedy)
            names.append(f"epsilon_{epsilon}")

    plot_result(solvers, names)

def decaying_epsilon_greedy_run(
        t: int = 5000,
        decaying_epsilon: Callable[[int], int] = lambda t : 1 / t,
        n: int = 10
) -> None:
    bernoulli_bandit_n = BernoulliBandit(n)

    decaying_epsilon_greedy = DecayingEpsilonGreedy(bernoulli_bandit_n, decaying_epsilon)
    decaying_epsilon_greedy.run(t)
    plot_result([decaying_epsilon_greedy], ["decaying_epsilon_greedy"])

def ucb_run(coef: float = 1.0, t: int = 5000, n: int = 10) -> None:
    bernoulli_bandit_n = BernoulliBandit(n)

    ucb = UCB(bernoulli_bandit_n, coef)
    ucb.run(t)
    plot_result([ucb], ["ucb"])

def thompson_sampling_run(t: int = 5000, n: int = 10) -> None:
    bernoulli_bandit_n = BernoulliBandit(n)

    thompson_sampling = ThompsonSampling(bernoulli_bandit_n)
    thompson_sampling.run(t)
    plot_result([thompson_sampling], ["thompson_sampling"])

if __name__ == '__main__':
    np.random.seed(2)
    epsilon_greedy_run()

    np.random.seed(2)
    epsilon_greedy_run([1e-4, 0.01, 0.1, 0.25, 0.5])

    np.random.seed(0)
    decaying_epsilon_greedy_run()

    # np.random.seed(1)
    # decaying_epsilon_greedy_run(5000, lambda t : 1 / np.power(t, 0.8))

    np.random.seed(1)
    ucb_run()

    np.random.seed(1)
    thompson_sampling_run(50000)




