import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Setup
methods = ['Ours', 'SOTA', 'VDO', 'Dyna']

# Data Structure: {Method: {Speed: [Score, Recall, FP]}}
data = {
    'Ours': {
        0: [0.7908, 0.7919, 0.0011],
        1: [0.7816, 0.4762, 0.0007],
        2: [0.3991, 0.4082, 0.0398]
    },
    'SOTA': {
        0: [0.4390, 0.4392, 0.0001],
        1: [0.5793, 0.2733, 0.0001],
        2: [0.1992, 0.1925, 0.0267]
    },
    'VDO': {
        0: [0.6635, 0.6638, 0.0003],
        1: [0.5455, 0.4996, 0.2603],
        2: [0.0072, 0.7770, 1.3568]
    },
    'Dyna': {
        0: [0.4282, 0.4284, 0.0002],
        1: [0.6163, 0.3113, 0.0011],
        2: [0.3173, 0.3067, 0.0197]
    }
}

# Metrics
metrics = ['Score', 'Mean Recall', 'Mean False Positive']

# Function to interpolate for speed 1.5
def get_interpolated_value(y0, y1, y2, target_x=1.5):
    # Points: (0, y0), (1, y1), (2, y2)
    # Fit polynomial degree 2
    x = [0, 1, 2]
    y = [y0, y1, y2]
    coeffs = np.polyfit(x, y, 2)
    p = np.poly1d(coeffs)
    return p(target_x)

# Prepare data for plotting
# X-axis for plotting: 0, 1, 2, 3
# Corresponding to Data Speeds: 0, 1, 1.5, 2
plot_data = {method: {metric: [] for metric in metrics} for method in methods}

for method in methods:
    for i, metric in enumerate(metrics):
        y0 = data[method][0][i]
        y1 = data[method][1][i]
        y2 = data[method][2][i]
        
        y_1_5 = get_interpolated_value(y0, y1, y2, 1.5)
        
        # Plotting values: [Speed 0, Speed 1, Speed 1.5, Speed 2]
        # Mapped to X: [0, 1, 2, 3]
        plot_data[method][metric] = [y0, y1, y_1_5, y2]

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x_axis = [0, 1, 2, 3]
x_labels = ['0', '1', '2', '3'] # As requested: 1.5->2, 2->3. Implicitly speed 0->0, 1->1.

markers = ['o', 's', '^', 'D']
colors = ['r', 'b', 'g', 'm']
linestyles = ['-', '--', '-.', ':']

for i, metric in enumerate(metrics):
    ax = axes[i]
    for j, method in enumerate(methods):
        y_values = plot_data[method][metric]
        ax.plot(x_axis, y_values, marker=markers[j], label=method, color=colors[j], linestyle=linestyles[j])
    
    ax.set_title(metric)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_labels) # Though the values 0,1,2,3 correspond to the indices, setting labels explicitly ensures clarity if needed.
    ax.set_xlabel('Speed Setting (Mapped)')
    ax.set_ylabel(metric)
    ax.grid(True, linestyle='--', alpha=0.6)
    if i == 0: # Legend on the first plot or all? Let's put on first for now or commonly inside
         ax.legend()

plt.tight_layout()
plt.savefig('performance_plots.png')

# Output the interpolated values for user reference
interpolated_results = {}
for method in methods:
    interpolated_results[method] = {
        'Score': plot_data[method]['Score'][2],
        'Recall': plot_data[method]['Mean Recall'][2],
        'FP': plot_data[method]['Mean False Positive'][2]
    }

print("Interpolated values (Speed 1.5 mapped to 2):")
print(interpolated_results)