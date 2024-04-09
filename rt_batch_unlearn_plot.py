import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    "Sample Sizes": ["2", "8", "32", "128", "512", "1024", "2048"],
    "RT": [72, 72, 116, 176, 176, 176, 176],
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
plt.figure(figsize=(8, 6))
sns.barplot(x="Sample Sizes", y="RT", data=df)

plt.ylim(50, 190)

# Annotate each bar with its value
for index, row in df.iterrows():
    plt.text(index, row.RT + 2, row.RT, color="black", ha="center")

plt.title("RT for Batch Unlearning Models", pad=20)
plt.xlabel("Sample Sizes")
plt.ylabel("RT", labelpad=10)
# plt.xticks(
#     rotation=45
# )  # Optional: Rotate labels if they overlap or for better readability

plt.tight_layout()
plt.savefig("rt_batch_unlearn.pgf", bbox_inches="tight")
plt.show()
