from matplotlib import pyplot as plt
import pandas


results = pandas.read_csv('result.csv', header=0)

print('Columns:')
print(results.columns)
print()

top_results = results.sort_values(by=[' mean_score'], ascending=False).head() 
print('Top score:')
print(top_results)

for hidden_size in [20, 30, 40]:

    results_with_specific_hidden_size = results[results[' hidden_layer_size'] == hidden_size]
    sorted_by_num_of_features = results_with_specific_hidden_size.sort_values(by='num_of_features')
    plt.plot(sorted_by_num_of_features['num_of_features'], sorted_by_num_of_features[' mean_score'])

plt.show()

# print(results)
# print(results.sort_values(by='mean_score'))



