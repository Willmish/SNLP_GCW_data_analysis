import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import matplotlib

# Update matplotlib settings for LaTeX compatibility
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
    }
)
# matplotlib.use("pgf")

# Setting seaborn style and matplotlib parameters for aesthetics
sns.set(style="whitegrid", font_scale=1.25)
plt.rcParams.update(
    {
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.linewidth": 1.5,
        "xtick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
    }
)

# Define the base directory and unlearning type
base_dir = "eval_combined"
unlearning_type = "sequential_unlearning"

# Use glob to find all CSV files matching the specified pattern
csv_files = glob.glob(
    os.path.join(base_dir, f"{unlearning_type}/**/results.csv"), recursive=True
)


df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    # Extract the group name, batch number, and sample size from the directory name
    group_name = os.path.basename(os.path.dirname(file))
    temp_df["Group"] = group_name
    num_split = int(group_name.split("-")[1])
    sample_count = int(group_name.split("-")[2])
    temp_df["sample_count"] = sample_count
    temp_df["num_split"] = num_split
    temp_df["epoch_num"] = [
        0,
        20,
    ]  # for seq, we only have epoch 0 and last epoch num (20 hardcoded here)
    df_list.append(temp_df)

# Concatenate all dataframes into one
combined_df = pd.concat(df_list, ignore_index=True)

# Sort the dataframe as needed
combined_df.sort_values(by=["num_split", "epoch_num"], inplace=True)

# Melt the dataframe for seaborn plotting
combined_df_melted = combined_df.melt(
    id_vars=["Group", "num_split", "epoch_num"],
    var_name="Benchmark",
    value_name="Accuracy",
)

# Remove specific benchmarks if necessary
combined_df_melted = combined_df_melted[
    combined_df_melted["Benchmark"].str.contains("truthfulqa_mc2_acc") == False
]

# Additional processing for benchmark names
combined_df_melted["Benchmark"] = combined_df_melted["Benchmark"].replace(
    "safety_eval_beaverdam-7b", "beaverdam-7b"
)
combined_df_melted["Benchmark"] = combined_df_melted["Benchmark"].str.replace(
    "_", r"\_"
)
combined_df_melted["Benchmark"] = combined_df_melted["Benchmark"].str.replace(
    "%", r"\%"
)

# Extract batch size and sample size
combined_df_melted["num_split"] = combined_df_melted["Group"].apply(
    lambda x: int(x.split("-")[1])
)
combined_df_melted["sample_count"] = combined_df_melted["Group"].apply(
    lambda x: int(x.split("-")[2])
)

# Set up the subplot grid
fig, axs = plt.subplots(3, 3, figsize=(20, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
sample_counts = [128, 512, 1024]
num_splits = [4, 16, 64]

# Plotting loop
for i, sample_count in enumerate(sample_counts):
    for j, num_split in enumerate(num_splits):
        ax = axs[i, j]
        df_plot = combined_df_melted[
            (combined_df_melted["sample_count"] == sample_count)
            & (combined_df_melted["num_split"] == num_split)
        ]
        sns.lineplot(
            x="epoch_num",
            y="Accuracy",
            hue="Benchmark",
            style="Benchmark",
            data=df_plot,
            markers=True,
            dashes=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"Number of Split: {num_split}, Number of Samples: {sample_count}")
        ax.set_xlabel("Epoch Number")
        ax.set_ylabel(r"acc/norm\_acc", labelpad=10)
        ax.set_ylim(0.2, 1.0)  # Adjust as needed
        # Customize each subplot as needed (e.g., setting tick parameters, legend, etc.)

# Adjust figure layout and titles as needed
plt.tight_layout()
# plt.savefig("con_unlearn_acc_idx_subplots.pgf", bbox_inches="tight")
plt.savefig("con_unlearn_acc_idx_subplots.png", bbox_inches="tight")
plt.show()


# for row_index, sample_count in enumerate(sample_counts):
#     # Create a new figure for the current row
#     fig, axs = plt.subplots(
#         1, len(num_splits), figsize=(20, 4)
#     )  # Adjust figsize as needed

#     for col_index, num_split in enumerate(num_splits):
#         ax = axs[col_index]  # Access the subplot for the current column
#         df_plot = combined_df_melted[
#             (combined_df_melted["sample_count"] == sample_count)
#             & (combined_df_melted["num_split"] == num_split)
#         ]
#         sns.lineplot(
#             x="epoch_num",
#             y="Accuracy",
#             hue="Benchmark",
#             style="Benchmark",
#             data=df_plot,
#             markers=True,
#             dashes=False,
#             legend=False,
#             ax=ax,
#         )
#         ax.set_title(f"Number of Split: {num_split}, Number of Samples: {sample_count}")
#         # ax.set_title(f"Number of Split: {num_split}")
#         ax.set_xlabel("Epoch Number")
#         ax.set_ylabel(r"acc/norm\_acc", labelpad=10)
#         ax.set_ylim(0.2, 1.0)  # Adjust as needed

#     # Adjust layout
#     plt.tight_layout()
#     # Save the current row of subplots
#     plt.savefig(
#         f"seq_unlearn_acc_epoch_subplots_sample_num_{sample_count}.pgf",
#         bbox_inches="tight",
#     )
#     plt.show()
# plt.close()  # Close the figure to free memory
