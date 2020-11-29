import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from tabulate import tabulate

from run_experiment import get_classifiers

clfs = get_classifiers()
scores = np.load("results.npy")

fig, ax = plt.subplots()

current_num_of_features = 1
current_average_score = []
average_scores = []

for score, classifier in zip(scores, clfs):
    _, num_of_features, _, _, hidden_neurons, _, _, momentum = classifier.split('_')

    num_of_features = int(num_of_features)
    hidden_neurons = int(hidden_neurons)
    momentum = float(momentum)

    if num_of_features != current_num_of_features:
        average_scores.append(np.mean(current_average_score))
        current_average_score = []
        current_num_of_features += 1

    current_average_score.append(score)
    
# print(average_scores)

plt.plot(average_scores)
plt.show()

exit()

# ALL AVERAGE

fig, ax = plt.subplots()

current_num_of_features = 1
current_average_score = []
average_scores = []

for score, classifier in zip(scores, clfs):
    _, num_of_features, _, _, hidden_neurons, _, _, momentum = classifier.split('_')

    num_of_features = int(num_of_features)
    hidden_neurons = int(hidden_neurons)
    momentum = float(momentum)

    if num_of_features != current_num_of_features:
        average_scores.append(np.mean(current_average_score))
        current_average_score = []
        current_num_of_features += 1

    current_average_score.append(score)
    
# print(average_scores)

plt.plot(average_scores)
plt.show()

exit()

fig, ax = plt.subplots()
for score, classifier in zip(scores, clfs):
    color = "r"
    marker = "o"

    # if 'features_1_' in classifier \
    #     or 'features_2_' in classifier \
    #     or 'features_3_' in classifier \
    #     or 'features_4_' in classifier \
    #     or 'features_5_' in classifier \
    #     or 'features_6_' in classifier \
    #     or 'features_7_' in classifier \
    #     or 'features_8_' in classifier \
    #     or 'features_9_' in classifier: \
    #     color = 'r'
    # elif 'features_1' in classifier:
    #     color = 'r'
    # elif 'features_2' in classifier or 'features_3' in classifier:
    #     color = 'orange'
    # elif 'features_4' in classifier or 'features_5' in classifier:
    #     color = 'g'
    # elif 'features_6' in classifier:
    #     color = 'b'

    if "hidden_20" in classifier:
        color = "b"
    if "hidden_50" in classifier:
        color = "g"
    if "hidden_90" in classifier:
        color = "y"

    # if 'momentum_0.0' in classifier:
    #     marker = 'o'
    # else:
    #     marker = 'X'

    mean = np.mean(score)
    std = np.std(score)

    ax.scatter(mean, std, c=color, marker=marker)


legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="20 ukrytych w warstwie ukrytej",
        markerfacecolor="b",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="50 neuronów w warstwie ukrytej",
        markerfacecolor="g",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="90 neuronów w warstwie ukrytej",
        markerfacecolor="y",
        markersize=10,
    ),
]


ax.legend(handles=legend_elements)

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
headers = [key for key in clfs.keys()]
names_column = np.array([[key] for key in clfs.keys()])

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(
    np.concatenate((names_column, significance), axis=1), headers
)
# print("Statistical significance (alpha = 0.05):\n", significance_table)


stat_better = significance * advantage
stat_better_table = tabulate(
    np.concatenate((names_column, stat_better), axis=1), headers
)
print("Statistically significantly better:\n", stat_better_table)
