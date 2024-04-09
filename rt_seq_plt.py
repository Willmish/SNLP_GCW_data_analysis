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

sns.set(style="whitegrid", font_scale=1.8)
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

# Defining the x-axis labels (splits) and the RT values for each $|D_u|$
splits = ["4 Splits", "16 Splits", "64 Splits"]
rt_du_128 = [368, 372, 256]
rt_du_512 = [248, 252, 372]
rt_du_1024 = [248, 176, 256]

# Setting the positions of the bars on the x-axis
x = np.arange(len(splits))
width = 0.2  # the width of the bars

# Creating the bars
fig, ax = plt.subplots(figsize=(10, 7))
rects1 = ax.bar(x - width, rt_du_128, width, label="$|D_u| = 128$")
rects2 = ax.bar(x, rt_du_512, width, label="$|D_u|=512$")
rects3 = ax.bar(x + width, rt_du_1024, width, label="$|D_u|=1024$")

# Adding some text for labels, title and custom x-axis tick labels, etc.
plt.title("RT for Sequential Unlearning", pad=20)
plt.ylabel("RT", labelpad=10)
# ax.set_ylabel("RT", label)
# ax.set_title("RT for Sequential Unlearning")
ax.set_xticks(x)
ax.set_xticklabels(splits)

plt.legend(
    # title="Method",
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.08,
    ),  # The negative value in the second argument pushes the legend below the plot.
    ncol=3,  # Adjust based on your number of legend items
    frameon=True,
    fancybox=True,
    # borderpad=1.25,
    edgecolor="black",
)


# Function to automatically label each bar with its height value
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


# # Apply the function to each set of bars
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

plt.tight_layout()
plt.savefig("rt_seq.pgf", bbox_inches="tight")
plt.show()
