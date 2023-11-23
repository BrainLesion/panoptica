import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of CSV files
csv_list = [
    "benchmark/m1max/dataframe.csv",
    "benchmark/ryzen9/dataframe.csv",
]

# Use list comprehension to read and concatenate CSV files
concatenated_df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)

# Set custom colors
colors = ["#1AFF1A", "#4B0092"]

title_font_size = 16
label_font_size = 14
tick_font_size = 12

# Get unique platforms and conditions
unique_platforms = concatenated_df["platform"].unique()
unique_conditions = concatenated_df["condition"].unique()

# Create subplots for each condition
fig, axes = plt.subplots(nrows=len(unique_conditions), ncols=3, figsize=(18, 6 * len(unique_conditions)))

for i, condition in enumerate(unique_conditions):
    # Filter data for the current condition
    condition_df = concatenated_df[concatenated_df["condition"] == condition]
    
    # Boxplot for Approximation Time
    sns.boxplot(
        x="platform",
        y="approximation",
        data=condition_df,
        notch=False,
        ax=axes[i, 0],
        palette=dict(zip(unique_platforms, colors)),
    )
    axes[i, 0].set_title(f"Approximation Time Comparison - {condition}", fontsize=title_font_size)
    axes[i, 0].set_xlabel("Platform", fontsize=label_font_size)
    axes[i, 0].set_ylabel("Approximation Time (s)", fontsize=label_font_size)
    axes[i, 0].tick_params(axis="both", which="major", labelsize=tick_font_size)
    
    # Boxplot for Matching Time
    sns.boxplot(
        x="platform",
        y="matching",
        data=condition_df,
        notch=False,
        ax=axes[i, 1],
        palette=dict(zip(unique_platforms, colors)),
    )
    axes[i, 1].set_title(f"Matching Time Comparison - {condition}", fontsize=title_font_size)
    axes[i, 1].set_xlabel("Platform", fontsize=label_font_size)
    axes[i, 1].set_ylabel("Matching Time (s)", fontsize=label_font_size)
    axes[i, 1].tick_params(axis="both", which="major", labelsize=tick_font_size)
    
    # Boxplot for Evaluation Time
    sns.boxplot(
        x="platform",
        y="evaluation",
        data=condition_df,
        notch=False,
        ax=axes[i, 2],
        palette=dict(zip(unique_platforms, colors)),
    )
    axes[i, 2].set_title(f"Evaluation Time Comparison - {condition}", fontsize=title_font_size)
    axes[i, 2].set_xlabel("Platform", fontsize=label_font_size)
    axes[i, 2].set_ylabel("Evaluation Time (s)", fontsize=label_font_size)
    axes[i, 2].tick_params(axis="both", which="major", labelsize=tick_font_size)

# Save the entire figure after all subplots have been created
plt.tight_layout()
plt.savefig("all_subplots.png", bbox_inches="tight")
plt.close()

print("Plots saved successfully.")
