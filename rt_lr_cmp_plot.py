import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
    }
)
matplotlib.use("pgf")

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

# Sample sizes and corresponding RT values for Normal LR and Scaled LR
sample_sizes = ["32", "128", "512", "1024", "2048"]
normal_lr_rt = [116, 176, 176, 176, 176]
scaled_lr_rt = [248, 372, 468, 468, 956]

# Positions of the bars on the x-axis
r1 = np.arange(len(normal_lr_rt))
r2 = [x + 0.2 for x in r1]

# Create the bars
plt.figure(figsize=(8, 6))
plt.bar(r1, normal_lr_rt, width=0.2, label="Normal LR")
plt.bar(r2, scaled_lr_rt, width=0.2, label="Scaled LR")

# Add labels, title, and legend
plt.xlabel("Sample Sizes")
plt.ylabel("RT", labelpad=10)
plt.xticks([r + 0.1 for r in range(len(normal_lr_rt))], sample_sizes)
plt.title("RT with Constant and Scaled Learning Rate", pad=20)
plt.legend()

# Text for labels, title, and custom x-axis tick labels
# for i in range(len(r1)):
#     plt.text(r1[i], normal_lr_rt[i] + 20, str(normal_lr_rt[i]), ha="center")
#     plt.text(r2[i], scaled_lr_rt[i] + 20, str(scaled_lr_rt[i]), ha="center")

plt.tight_layout()
plt.savefig("rt_lr_cmp.pgf", bbox_inches="tight")
plt.show()
