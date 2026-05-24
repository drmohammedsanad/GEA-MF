import matplotlib.pyplot as plt

# Full dataset (mean values)
data = {
    'ML-100K': {
        'Popularity': (0.0877, 11.2128),
        'MF': (0.0851, 11.0863),
        'SVD++': (0.0853, 11.0835),
        'BPR': (0.1326, 11.1030),
        'GEA-MF': (0.1432, 11.1191)
    },
    'ML-1M': {
        'Popularity': (0.0601, 6.1134),
        'MF': (0.0042, 4.2621),
        'SVD++': (0.0040, 5.0822),
        'BPR': (0.0092, 1.4223),
        'GEA-MF': (0.0087, 0.8779)
    },
    'LastFM': {
        'Popularity': (0.0005, 2.9206),
        'MF': (0.0010, 0.0203),
        'SVD++': (0.0010, 0.0434),
        'BPR': (0.0081, 0.0942),
        'GEA-MF': (0.0084, 0.1230)
    }
}

# Colors for datasets
colors = {
    'ML-100K': 'blue',
    'ML-1M': 'green',
    'LastFM': 'red'
}

# Markers for models
markers = {
    'Popularity': 's',
    'MF': 'x',
    'SVD++': 'D',
    'BPR': 'o',
    'GEA-MF': '^'
}

plt.figure(figsize=(6,5))

# Plot all points
for dataset, models in data.items():
    for model, (recall, exposure) in models.items():
        plt.scatter(
            recall,
            exposure,
            color=colors[dataset],
            marker=markers[model],
            s=80,
            label=f"{dataset} - {model}"
        )

# Remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))

plt.legend(unique.values(), unique.keys(), fontsize=7, loc='best')

plt.xlabel('Recall (higher is better)')
plt.ylabel('Exposure (lower is better)')
plt.title('Accuracy vs Fairness Trade-off (All Models)')

plt.grid()

plt.savefig('accuracy_fairness_all_models.png', dpi=300)
plt.show()