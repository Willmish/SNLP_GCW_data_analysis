import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.preprocessing import MinMaxScaler

# Define file paths
base_path = 'eval_mink/'
badloss_files = [
    f'{base_path}continuous_unlearning_42_badloss_mink.csv',
    f'{base_path}continuous_unlearning_456_badloss_mink.csv',
    f'{base_path}continuous_unlearning_1234_badloss_mink.csv',
    f'{base_path}continuous_unlearning_8888_badloss_mink.csv',
    f'{base_path}continuous_unlearning_114514_badloss_mink.csv',
]
safety_files = [
    f'{base_path}continuous_unlearning_42_safety.csv',
    f'{base_path}continuous_unlearning_456_safety.csv',
    f'{base_path}continuous_unlearning_1234_safety.csv',
    f'{base_path}continuous_unlearning_8888_safety.csv',
    f'{base_path}continuous_unlearning_114514_safety.csv',
]

# Load CSVs
badloss_dfs = [pd.read_csv(file) for file in badloss_files]
safety_dfs = [pd.read_csv(file) for file in safety_files]

# Combine and compute average for each step
badloss_combined = pd.concat(badloss_dfs).groupby('Step').mean().reset_index()
safety_combined = pd.concat(safety_dfs).groupby('Step').mean().reset_index()

# Apply smoothing
badloss_combined['bad loss'] = badloss_combined['bad loss'].rolling(window=10).mean()
badloss_combined['ratio mink unlearning/reference'] = badloss_combined['ratio mink unlearning/reference'].rolling(window=10).mean()

# Normalize smoothed data
scaler = MinMaxScaler()
columns_to_normalize = ['bad loss', 'ratio mink unlearning/reference']
badloss_combined[columns_to_normalize] = scaler.fit_transform(badloss_combined[columns_to_normalize])

# Plotting
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
    }
)
matplotlib.use("pgf")
plt.rcParams.update(
    {
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.linewidth": 1.5,  # Set the width of the axes border
        "xtick.major.size": 5,  # Set the length of the major ticks on x-axis
        "xtick.minor.size": 3,  # Set the length of the minor ticks on x-axis
        "ytick.major.size": 5,  # Set the length of the major ticks on y-axis
        "ytick.minor.size": 3,  # Set the length of the minor ticks on y-axis
    }
)

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Step')
ax1.set_ylabel('Smoothed then Normalized Bad Loss & Ratio Mink Unlearning/Reference', color=color)
ax1.plot(badloss_combined['Step'], badloss_combined['bad loss'], color=color, label='Normalized Bad Loss')
ax1.plot(badloss_combined['Step'], badloss_combined['ratio mink unlearning/reference'], color='tab:orange', linestyle='--', label='Normalized Ratio Mink Unlearning/Reference')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Safety Eval (clipped 0.7-0.9)', color=color)
ax2.plot(safety_combined['Step'], safety_combined['safety_eval_beaverdam-7b'], color=color, linestyle=':', label='Safety Eval')
ax2.tick_params(axis='y', labelcolor=color)

y_ticks_left = np.linspace(0, 1, num=6)  # For example, create 6 ticks from 0 to 1
y_ticks_right = np.linspace(0.7, 0.9, num=6)  # Ensure same number of ticks for right axis, but within 0.7 to 0.9
ax1.set_ylim([0, 1])  # Set the limit to match the ticks for the left axis
ax1.set_yticks(y_ticks_left)  # Apply the ticks to the left axis
ax2.set_ylim([0.7, 0.9])  # Set the limit to match the ticks for the right axis
ax2.set_yticks(y_ticks_right)  # Apply the ticks to the right axis

plt.legend(
    # title="Method",
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.2,
    ),  # The negative value in the second argument pushes the legend below the plot.
    ncol=3,  # Adjust based on your number of legend items
    frameon=True,
    fancybox=True,
    # borderpad=1.25,
    edgecolor="black",
)

fig.tight_layout()
plt.title('Smoothed then Normalized Metrics over Steps (Safety Eval Clipped 0.7-0.9)')
#plt.savefig(os.path.join(base_path, "badloss_mink_safety_vs_step.png"))
plt.savefig(os.path.join(base_path, "badloss_mink_safety_vs_step.pgf"), bbox_inches="tight")

