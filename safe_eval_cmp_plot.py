import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
    }
)
matplotlib.use("pgf")

# Data
data = {
    "Samples in the unlearn set": [128, 512, 1024] * 4,
    "Ratio of safe responses (%)": [
        77.85714285714286,
        77.42857142857142,
        78.14285714285714,  # Batch
        81.57142857142857,
        85.28571428571429,
        88.42857142857143,  # Sequential 4 splits
        79.42857142857143,
        90.57142857142857,
        90.42857142857142,  # Sequential 16 splits
        93,
        98.42857142857143,
        99.14285714285714,  # Sequential 64 splits
    ],
    "Method": ["Batch"] * 3
    + ["Seq. 4 splits"] * 3
    + ["Seq. 16 splits"] * 3
    + ["Seq. 64 splits"] * 3,
}

df = pd.DataFrame(data)

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

# Plot
plt.figure(figsize=(10, 8))
sns.lineplot(
    data=df,
    x="Samples in the unlearn set",
    y="Ratio of safe responses (%)",
    hue="Method",
    style="Method",
    markers=True,
    dashes=False,
    markersize=15,
)

# Continuous Unlearning line
plt.axhline(
    y=89, color="green", linestyle="dotted", linewidth=2, label="Continuous unlearning"
)

plt.title("Safety Evaluation of Unlearning Methods", pad=20)
plt.xlabel("Samples in the unlearn set")
plt.ylabel(r"Ratio of safe responses (\%)", labelpad=10)
plt.xlim(100, 1050)
plt.ylim(75, 100)
plt.xticks([128, 512, 1024])
plt.yticks([75, 80, 85, 90, 95, 100])

ax = plt.gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color("black")

# plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")

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

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("safe_eval_cmp_plot.pgf", bbox_inches="tight")
plt.show()
