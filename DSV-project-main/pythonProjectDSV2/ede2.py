import numpy as np
from sklearn import datasets
print("Starting script...")
iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names
data_list = data.tolist()
print("Data loaded and converted to list of lists.")
means = np.mean(data, axis=0)
std_devs = np.std(data, axis=0)
print("Overall Mean and Standard Deviation for each measurement column:")
for i, col_name in enumerate(iris.feature_names):
    print(f"{col_name}: Mean = {means[i]:.2f}, Std Dev = {std_devs[i]:.2f}")
species_data = {name: [] for name in target_names}
for i, species in enumerate(target):
    species_name = target_names[species]
    species_data[species_name].append(data_list[i])

print("\nMean and Standard Deviation for each measurement column, by species:")

for species_name, measurements in species_data.items():
    measurements = np.array(measurements)
    species_means = np.mean(measurements, axis=0)
    species_std_devs = np.std(measurements, axis=0)
    print(f"\n{species_name.capitalize()}:")
    for i, col_name in enumerate(iris.feature_names):
        print(f"{col_name}: Mean = {species_means[i]:.2f}, Std Dev = {species_std_devs[i]:.2f}")

print("\nAnalysis to determine the 'best' measurement for guessing the species:")
for i, col_name in enumerate(iris.feature_names):
    species_ranges = []
    for species_name, measurements in species_data.items():
        species_means = np.mean(measurements, axis=0)
        species_std_devs = np.std(measurements, axis=0)
        species_ranges.append((species_means[i] - species_std_devs[i], species_means[i] + species_std_devs[i]))
    min_range = min([r[0] for r in species_ranges])
    max_range = max([r[1] for r in species_ranges])
    range_span = max_range - min_range
    print(f"{col_name}: Range Span = {range_span:.2f}")

print("Script completed.")