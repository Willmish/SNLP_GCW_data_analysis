import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import glob
import numpy as np

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
    }
)
matplotlib.use("pgf")


# Base directory containing the CSV files
base_dir = "eval_combined"

unlearning_type = "llm_unlearning_reproduced"
# Use glob to find all results.csv files within the base directory
csv_files = glob.glob(
    os.path.join(base_dir, f"{unlearning_type}/**/results.csv"), recursive=True
)

# Read each CSV file and store it in a list, including the path as part of the DataFrame for grouping
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    # Extract meaningful grouping information from file paths here, for example:
    group_name = os.path.basename(os.path.dirname(file))
    temp_df["Group"] = group_name
    # Extract the batch number from the group name
    seed_name = int(group_name.split("-")[2][4:])
    temp_df["Seed_Number"] = seed_name
    temp_df["idx_number"] = np.arange(1001, step=100)
    df_list.append(temp_df)  # Only taking last index

# Concatenate all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True)

# Sort the DataFrame by batch number
combined_df.sort_values(by=["Seed_Number", "idx_number"], inplace=True)
# print(combined_df)
# print(combined_df)
# Assuming your CSVs have a similar structure and you want to melt for seaborn
combined_df_melted = combined_df.melt(
    id_vars=["Group", "Seed_Number", "idx_number"],
    var_name="Benchmark",
    value_name="Accuracy",
)

# REMOVE TRUTHFULQA
combined_df_melted = combined_df_melted[
    combined_df_melted["Benchmark"].str.contains("truthfulqa_mc2_acc") == False
]

combined_df_melted["Benchmark"] = combined_df_melted["Benchmark"].replace(
    "safety_eval_beaverdam-7b", "beaverdam-7b"
)

combined_df_melted["Benchmark"] = combined_df_melted["Benchmark"].str.replace(
    "_", r"\_"
)
combined_df_melted["Benchmark"] = combined_df_melted["Benchmark"].str.replace(
    "%", r"\%"
)

# Plotting
sns.set(style="whitegrid", font_scale=1.75)
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


plt.figure(figsize=(8, 6))
plt.ylim(0.2, 1)
lineplot = sns.lineplot(
    x="idx_number",
    y="Accuracy",
    hue="Benchmark",
    style="Benchmark",
    data=combined_df_melted,
    markers=True,
    dashes=False,
    legend="brief",
    markersize=10,
)

ax = plt.gca()
idx_values = sorted(combined_df_melted["idx_number"].unique())
ax.set_xticks(idx_values)
samle_counts = [2 * idx for idx in idx_values]
ax.set_xticklabels(samle_counts)

plt.title("Continuous Unlearning", pad=20)
# plt.xlabel("Index Number")
plt.xlabel("Sample Count")
plt.ylabel(r"acc/norm\_acc", labelpad=10)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.187,
    ),  # The negative value in the second argument pushes the legend below the plot.
    ncol=2,  # Adjust based on your number of legend items
    frameon=True,
    fancybox=True,
    borderpad=1.25,
    edgecolor="black",
)

ax.legend().set_visible(False)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color("black")


plt.tight_layout()
# Get handles and labels from the current figure
handles, labels = plt.gca().get_legend_handles_labels()

plt.savefig("con_unlearn_acc_idx.pgf", bbox_inches="tight")
plt.show()
