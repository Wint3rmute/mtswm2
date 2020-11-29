import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate
from matplotlib import pyplot as plt

from run_experiment import get_classifiers

clfs = get_classifiers()

scores = np.load("results.npy")
# print("Folds:\n", scores)


for score in scores:
    mean = np.mean(score)
    std = np.std(score)

    plt.scatter(mean, std)

plt.show()
exit()

alfa = 0.05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)


# print(clfs.keys())
headers = [ key for key in clfs.keys() ]
names_column = np.array([[ key ] for key in clfs.keys()])

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table)



stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)