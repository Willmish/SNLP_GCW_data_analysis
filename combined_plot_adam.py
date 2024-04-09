import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import glob
import numpy as np

# Configure matplotlib for LaTeX integration
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": True,
        "font.serif": ["Times"],
    }
)
matplotlib.use("pgf")

# Base directory containing the CSV files
base_dir = "eval_combined"

# Setup for plot 1 (llm_unlearning_reproduced)
unlearning_type_1 = "llm_unlearning_reproduced"
csv_files_1 = glob.glob(
    os.path.join(base_dir, f"{unlearning_type_1}/**/results.csv"), recursive=True
)
df_list_1 = []
for file in csv_files_1:
    temp_df = pd.read_csv(file)
    group_name = os.path.basename(os.path.dirname(file))
    temp_df["Group"] = group_name
    seed_name = int(group_name.split("-")[2][4:])
    temp_df["Seed_Number"] = seed_name
    temp_df["idx_number"] = np.arange(1001, step=100)
    df_list_1.append(temp_df)
combined_df_1 = pd.concat(df_list_1, ignore_index=True)
combined_df_1.sort_values(by=["Seed_Number", "idx_number"], inplace=True)
combined_df_melted_1 = combined_df_1.melt(
    id_vars=["Group", "Seed_Number", "idx_number"],
    var_name="Benchmark",
    value_name="Accuracy",
)

# Setup for plot 2 (batch_unlearning)
unlearning_type_2 = "batch_unlearning"
csv_files_2 = glob.glob(
    os.path.join(base_dir, f"{unlearning_type_2}/**/results.csv"), recursive=True
)
df_list_2 = []
for file in csv_files_2:
    temp_df = pd.read_csv(file)
    group_name = os.path.basename(os.path.dirname(file))
    temp_df["Group"] = group_name
    batch_number = int(group_name.split("-")[1])
    temp_df["Batch_Number"] = batch_number
    temp_df["epoch_num"] = np.arange(21, step=4)
    df_list_2.append(temp_df)
combined_df_2 = pd.concat(df_list_2, ignore_index=True)
combined_df_2.sort_values(by=["Batch_Number", "epoch_num"], inplace=True)
combined_df_melted_2 = combined_df_2.melt(
    id_vars=["Group", "Batch_Number", "epoch_num"],
    var_name="Benchmark",
    value_name="Accuracy",
)
combined_df_melted_2["Benchmark"] = (
    combined_df_melted_2["Benchmark"].str.replace("_", r"\_").str.replace("%", r"\%")
)

# Create a unified plotting figure
sns.set(style="whitegrid", font_scale=1.75)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharey=True)

# Plot 1
sns.lineplot(
    x="idx_number",
    y="Accuracy",
    hue="Benchmark",
    data=combined_df_melted_1,
    markers=True,
    dashes=False,
    ax=axs[0],
)
axs[0].set_title("LLM Unlearning Reproduced")
axs[0].set_xlabel("Index Number")
axs[0].set_ylabel("Accuracy")

# Plot 2
sns.lineplot(
    x="epoch_num",
    y="Accuracy",
    hue="Benchmark",
    style="Benchmark",
    data=combined_df_melted_2,
    markers=True,
    dashes=False,
    ax=axs[1],
    legend=False,  # Hide legend here to combine it later
)
axs[1].set_title("Batch Unlearning")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("")

# Adjust layout
plt.tight_layout()

# Add a legend outside the rightmost plot
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    fancybox=True,
    shadow=True,
    ncol=5,
)

# Save to PGF
plt.savefig("unlearning_results.pgf", bbox_inches="tight")
plt.show()
