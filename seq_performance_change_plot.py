import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
    }
)
matplotlib.use("pgf")

df = pd.read_csv("adam_seq_performance_change_data.csv")

df_melted = df.melt(
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

df_melted["Benchmark"] = df_melted["Benchmark"].replace(
    "safety_eval_beaverdam-7b", "beaverdam-7b"
)
# NAME TOO LONG

df_melted["Benchmark"] = df_melted["Benchmark"].str.replace("_", r"\_")
df_melted["Benchmark"] = df_melted["Benchmark"].str.replace("%", r"\%")

# Set the seaborn style
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

# Specify the benchmarks you want to emphasize or de-emphasize
emphasize_benchmarks = [
    r"truthfulqa\_mc2\_acc",
    r"hellaswag\_acc\_norm",
    # "safety_eval_beaverdam-7b",
    "beaverdam-7b",
]
deemphasize_benchmarks = [
    r"winogrande\_acc",
    r"arc\_challenge\_acc\_norm",
    r"mmlu\_acc",
    r"toxigen\_acc\_norm",
]

plt.figure(figsize=(10, 10))

# Highlighting one specific line by plotting it again with a higher alpha or different linewidth
sns.lineplot(
    x="split",
    y="Accuracy",
    hue="Benchmark",
    style="epoch_num",
    data=df_melted[df_melted["Benchmark"].isin(emphasize_benchmarks)],
    linewidth=2.5,
    alpha=1,
    markersize=15,
    markers=True,
    dashes=False,
    legend="brief",
    # legend=False,
)

# lineplot = sns.lineplot(
#     x="split",
#     y="Accuracy",
#     hue="Benchmark",
#     style="epoch_num",
#     data=df_melted[df_melted["Benchmark"].isin(deemphasize_benchmarks)],
#     markers=True,
#     markersize=10,
#     dashes=False,
#     alpha=0.3,
#     legend="brief",
#     # legend=False,
# )
# Your existing handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Desired order of benchmark labels
benchmark_labels = [
    r"truthfulqa\_mc2\_acc",
    r"hellaswag\_acc\_norm",
    # "safety\_eval\_beaverdam-7b",
    "beaverdam-7b",
    # "winogrande_acc",
    # "arc_challenge_acc_norm",
    # "mmlu_acc",
    # "toxigen_acc_norm",
]

# Desired order of epoch labels
epoch_labels = ["0", "20"]

benchmark_handles = []
benchmark_labels_ordered = []
epoch_handles = []
epoch_labels_ordered = []

# Filter the handles and labels for benchmarks
for label in benchmark_labels:
    if label in labels:
        index = labels.index(label)
        benchmark_handles.append(handles[index])
        benchmark_labels_ordered.append(label)

# Filter the handles and labels for epochs
for label in epoch_labels:
    if label in labels:
        index = labels.index(label)
        epoch_handles.append(handles[index])
        epoch_labels_ordered.append(f"Epoch {label}")

# Create the benchmark legend
benchmark_legend = plt.legend(
    benchmark_handles,
    benchmark_labels_ordered,
    title="Benchmark",
    bbox_to_anchor=(
        # 0.35,
        0.5,
        # -0.087,
        -0.1,
    ),  # The negative value in the second argument pushes the legend below the plot.
    loc="upper center",
    frameon=True,
    fancybox=True,
    edgecolor="black",
    ncol=3,
)

# Add the benchmark legend manually to the current Axes
plt.gca().add_artist(benchmark_legend)

# Create the epoch legend below the benchmark legend
epoch_legend = plt.legend(
    epoch_handles,
    epoch_labels_ordered,
    # title="Epoch Num",
    bbox_to_anchor=(
        # 0.85,
        0.5,
        # -0.087,
        -0.28,
    ),
    loc="upper center",
    frameon=True,
    fancybox=True,
    edgecolor="black",
    ncol=2,
)

plt.xticks(df_melted["split"].unique())
plt.title("Sequential Unlearning Sample Count 128", pad=20)
plt.xlabel("Split", labelpad=-5)
plt.ylabel(r"acc/norm\_acc", labelpad=10)

plt.ylim(0.35, 1)

plt.tight_layout()

plt.savefig(
    "seq_performance_change.pgf",
    bbox_extra_artists=(benchmark_legend, epoch_legend),
    bbox_inches="tight",
)
plt.show()
