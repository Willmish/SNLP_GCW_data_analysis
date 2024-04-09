import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
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
unlearning_type = "batch_unlearning"

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
    batch_number = int(group_name.split("-")[1])
    temp_df["Batch_Number"] = batch_number
    temp_df["epoch_num"] = np.arange(21, step=4)
    # df_list.append(temp_df[-1:])  # Only taking last index
    df_list.append(temp_df)

# Concatenate all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True)

# Sort the DataFrame by batch number
combined_df.sort_values(by=["Batch_Number", "epoch_num"], inplace=True)
# print(combined_df)
# print(combined_df)
# Assuming your CSVs have a similar structure and you want to melt for seaborn
combined_df_melted = combined_df.melt(
    id_vars=["Group", "Batch_Number", "epoch_num"],
    var_name="Benchmark",
    value_name="Accuracy",
)
# print(combined_df_melted["Benchmark"].unique())

# REMOVE TRUTHFULQA
combined_df_melted = combined_df_melted[
    combined_df_melted["Benchmark"].str.contains("truthfulqa_mc2_acc") == False
]

# print(combined_df_melted)

# combined_df_melted.to_csv("batch_unlearn_acc_epoch_plot_data.csv", index=False)
# grouped_stats.to_csv("batch_unlearn_acc_epoch_plot_data.csv", index=False)

combined_df_melted["Benchmark"] = combined_df_melted["Benchmark"].replace(
    "safety_eval_beaverdam-7b", "beaverdam-7b"
)
# NAME TOO LONG

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
    x="epoch_num",
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
epoch_values = sorted(combined_df_melted["epoch_num"].unique())
ax.set_xticks(epoch_values)


# # Highlighted version
# highlighted_benchmark = "safety_eval_beaverdam-7b"
# plt.figure(figsize=(20, 10))
# lineplot = sns.lineplot(
#     x="epoch_num",
#     y="Accuracy",
#     hue="Benchmark",
#     data=combined_df_melted[combined_df_melted["Benchmark"] != highlighted_benchmark],
#     markers=True,
#     dashes=False,
#     alpha=0.3,
# )

# # Highlighting one specific line by plotting it again with a higher alpha or different linewidth
# sns.lineplot(
#     x="epoch_num",
#     y="Accuracy",
#     data=combined_df_melted[combined_df_melted["Benchmark"] == highlighted_benchmark],
#     color="red",  # You can specify a color if you want
#     linewidth=3,  # Thicker line for the highlighted plot
#     alpha=1,  # Higher alpha for the highlighted plot
#     label=highlighted_benchmark,  # Ensure the label is correct for the legend
# )

# plt.xticks(rotation=45, ha="right")
plt.title("Batch Unlearning", pad=20)
plt.xlabel("Epoch")
plt.ylabel(r"acc/norm\_acc", labelpad=10)


# plt.legend(
#     title="Benchmark",
#     loc="upper center",
#     bbox_to_anchor=(
#         0.5,
#         -0.18,
#     ),  # The negative value in the second argument pushes the legend below the plot.
#     ncol=2,  # Adjust based on your number of legend items
#     frameon=True,
#     # mode="expand",
#     fancybox=True,
#     # borderaxespad=0.0,
# )

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

# for line in ax.get_xgridlines() + ax.get_ygridlines():
#     line.set_color("black")
# plt.show()

plt.tight_layout()
# Get handles and labels from the current figure
handles, labels = plt.gca().get_legend_handles_labels()

plt.savefig("batch_unlearn_acc_epoch.pgf", bbox_inches="tight")

# Create a new figure for the legend
plt.figure(figsize=(15, 1))
sns.set(style="whitegrid", font_scale=1.9)
plt.legend(
    handles,
    labels,
    # loc="upper center",
    ncol=6,
    frameon=True,
    fancybox=True,
    # borderpad=1.25,
    columnspacing=1.25,
    edgecolor="black",
)
plt.axis("off")  # Turn off the axis
plt.tight_layout()

# Save just the legend to a file
plt.show()
plt.savefig("legend.pgf", bbox_inches="tight")
# plt.close()  # Close the figure to avoid displaying it with plt.show()
