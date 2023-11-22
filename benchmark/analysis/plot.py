# import data
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


csv_list = [
    "benchmark/m1max/dataframe.csv",
    "benchmark/ryzen9/dataframe.csv",
]


# Use list comprehension to read and concatenate CSV files
concatenated_df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)

# Display the resulting DataFrame
print(concatenated_df)


# now we plot it!

# !!
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your concatenated DataFrame is named concatenated_df
# If not, replace it with the actual name of your DataFrame

# Example DataFrame structure:
# concatenated_df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)

# Set custom colors
colors = ["#1AFF1A", "#4B0092"]

title_font_size = 16
label_font_size = 14
tick_font_size = 12

# Create three separate subplots
fig, axes = plt.subplots(ncols=3, figsize=(18, 6))

# Boxplot for Approximation Time
sns.boxplot(
    x="platform",
    y="approximation",
    hue="platform",
    data=concatenated_df,
    notch=False,
    ax=axes[0],
    palette=dict(zip(concatenated_df["platform"].unique(), colors)),
)
axes[0].set_title("Approximation Time Comparison", fontsize=title_font_size)
axes[0].set_xlabel("", fontsize=label_font_size)
axes[0].set_ylabel("Approximation Time (s)", fontsize=label_font_size)
axes[0].tick_params(axis="both", which="major", labelsize=title_font_size)

# Boxplot for Matching Time
sns.boxplot(
    x="platform",
    y="matching",
    hue="platform",
    data=concatenated_df,
    notch=False,
    ax=axes[1],
    palette=dict(zip(concatenated_df["platform"].unique(), colors)),
)
axes[1].set_title("Matching Time Comparison", fontsize=title_font_size)
axes[1].set_xlabel("", fontsize=label_font_size)
axes[1].set_ylabel("Matching Time (s)", fontsize=label_font_size)
axes[1].tick_params(axis="both", which="major", labelsize=title_font_size)

# Boxplot for Evaluation Time
sns.boxplot(
    x="platform",
    y="evaluation",
    hue="platform",
    data=concatenated_df,
    notch=False,
    ax=axes[2],
    palette=dict(zip(concatenated_df["platform"].unique(), colors)),
)
axes[2].set_title("Evaluation Time Comparison", fontsize=title_font_size)
axes[2].set_xlabel("", fontsize=label_font_size)
axes[2].set_ylabel("Evaluation Time (s)", fontsize=label_font_size)
axes[2].tick_params(axis="both", which="major", labelsize=title_font_size)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
