import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure matplotlib to use LaTeX and set the font to serif
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
    }
)
matplotlib.use("pgf")

# Load the dataset
# sample_num = 128
# sample_num = 512
sample_num = 1024
df = pd.read_csv(f"adam_seq_performance_change_data_{sample_num}.csv")

# Only keep epochs 0 and 20
df_filtered = df[df["epoch_num"].isin([0, 20])]

# Melt the DataFrame for easier manipulation
df_melted = df_filtered.melt(
    id_vars=["epoch_num", "split", "sample_counts"],
    value_vars=[
        "winogrande_acc",
        "truthfulqa_mc2_acc",
        "hellaswag_acc_norm",
        "arc_challenge_acc_norm",
        "mmlu_acc",
        "toxigen_acc_norm",
        "safety_eval_beaverdam-7b",
    ],
    var_name="Benchmark",
    value_name="Accuracy",
)

# Normalize benchmark names
df_melted["Benchmark"] = df_melted["Benchmark"].replace(
    "safety_eval_beaverdam-7b", "beaverdam-7b"
)
df_melted["Benchmark"] = df_melted["Benchmark"].str.replace("_", " ")

# Pivot to get epochs as columns for each benchmark and split
df_pivot = df_melted.pivot_table(
    index=["Benchmark", "split"], columns="epoch_num", values="Accuracy"
)

# Calculate the difference between epoch 20 and epoch 0
df_pivot["diff"] = df_pivot[20] - df_pivot[0]

# Reset the index so we can use "Benchmark" and "split" as columns again
df_diff = df_pivot.reset_index()

# Set seaborn style
# sns.set(style="whitegrid", font_scale=1.2)

sns.set(style="whitegrid", font_scale=1.5)

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

# Initialize the plot
plt.figure(figsize=(9, 7))

# 0 line
plt.axhline(
    0, color="red", linewidth=2, linestyle="--"
)  # Draws a horizontal line at y=0

# Plotting each benchmark's accuracy difference across splits
sns.lineplot(
    data=df_diff,
    x="split",
    y="diff",
    hue="Benchmark",
    markers=True,
    dashes=False,
    style="Benchmark",
    linewidth=2,
    markersize=10,
)

plt.title(
    f"Benchmark Scores Difference across Splits\n Sample Count: {sample_num}", pad=20
)
plt.xlabel("Split")
plt.ylabel(r"Difference in Scores (Epoch 20 - Epoch 0)", labelpad=10)
# plt.legend(title="Benchmark", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(df_melted["split"].unique())

plt.legend(
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.17,
    ),  # The negative value in the second argument pushes the legend below the plot.
    ncol=3,  # Adjust based on your number of legend items
    frameon=True,
    fancybox=True,
    # borderpad=1.25,
    edgecolor="black",
)

ax = plt.gca()
# Emphasize the 0 on y-axis
# plt.axhline(
#     0, color="red", linewidth=2, linestyle="--"
# )  # Draws a horizontal line at y=0

# # Shade area below 0
# ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[0], color="grey", alpha=0.2)

# # Shade area above 0
# ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], color="blue", alpha=0.1)

plt.tight_layout()

# Save the plot
plt.savefig(
    f"seq_performance_change_v2_sample_num_{sample_num}.pgf", bbox_inches="tight"
)
plt.show()
