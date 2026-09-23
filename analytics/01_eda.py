# =====================================================================
# 01_eda.py  (BEGINNER VERSION)
# This does the same work as the advanced version, but uses simple
# loops and basic pandas commands instead of "clever" one-liners, so
# it is easier to read for someone new to Python/pandas.
# =====================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Make our charts look nice
sns.set_theme(style="whitegrid")

# Make pandas print wider tables in the terminal (just for readability)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# We will save all chart images inside a folder called "charts"
CHART_DIR = "charts"
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)


# =====================================================================
# TASK 1: Load the dataset ONE time, look at it, and save a backup CSV
# =====================================================================
print("=" * 70)
print("TASK 1: Loading the Titanic dataset")
print("=" * 70)

# This downloads the Titanic dataset the first time (needs internet),
# and after that it is cached on your computer.
df = sns.load_dataset("titanic")

print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() ---")
print(df.describe(include="all"))

print("\n--- df.shape ---")
print(df.shape)

# Find how many values are missing in each column, as a percentage
print("\n--- Missing values (%) per column ---")
total_rows = len(df)
for column_name in df.columns:
    missing_count = df[column_name].isna().sum()
    if missing_count > 0:
        missing_percent = (missing_count / total_rows) * 100
        print(column_name, ":", round(missing_percent, 2), "%")

# Save a copy of the raw data as a CSV file. This way, later steps
# (and 02_modeling.py) can just read this CSV instead of downloading
# the dataset again from the internet.
df.to_csv("titanic.csv", index=False)
print("\nSaved titanic.csv")


# =====================================================================
# TASK 2: Handle missing values, column by column
#   Rule:
#     - less than 5% missing  -> just drop those rows
#     - 5% to 30% missing     -> fill in (impute) the missing values
#     - more than 30% missing -> decide: drop the whole column,
#                                 or keep it and mark missing as its
#                                 own category
# =====================================================================
print("\n" + "=" * 70)
print("TASK 2: Cleaning missing values")
print("=" * 70)

# Read back from the CSV we just saved (not from the internet again)
df_clean = pd.read_csv("titanic.csv")

# Recalculate missing % on this fresh copy
missing_percent_dict = {}
for column_name in df_clean.columns:
    missing_count = df_clean[column_name].isna().sum()
    if missing_count > 0:
        missing_percent_dict[column_name] = (missing_count / len(df_clean)) * 100

print("Columns with missing values:")
for column_name, percent in missing_percent_dict.items():
    print(" ", column_name, ":", round(percent, 2), "%")

# --- Step A: "embarked" and "embark_town" have very little missing
# (less than 5%), so we simply drop those few rows.
for column_name in ["embarked", "embark_town"]:
    if column_name in df_clean.columns:
        rows_before = len(df_clean)
        df_clean = df_clean[df_clean[column_name].notna()]
        rows_after = len(df_clean)
        print("Dropped", rows_before - rows_after, "rows because '" +
              column_name + "' was missing there.")

# --- Step B: "age" is missing about 20% of the time (between 5-30%),
# so instead of dropping rows, we FILL IN the missing values.
# We use the median (middle value) because it is not affected much
# by very large or very small ages.
age_median_value = df_clean["age"].median()
df_clean["age"] = df_clean["age"].fillna(age_median_value)
print("Filled missing 'age' values with the median age:", age_median_value)

# --- Step C: "deck" is missing about 77% of the time (more than 30%).
# That is too much to reliably guess, so our decision is to DROP the
# whole column instead of guessing values for most rows.
if "deck" in df_clean.columns:
    df_clean = df_clean.drop(columns=["deck"])
    print("Dropped the whole 'deck' column (too much missing data).")

print("\nMissing values remaining after cleaning:")
print(df_clean.isna().sum())
print("New shape after cleaning:", df_clean.shape)


# =====================================================================
# TASK 3: Look at "age" and "fare" on their own (univariate analysis)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 3: Looking at age and fare")
print("=" * 70)

# Draw a histogram and a box plot for age and for fare
figure, axis_grid = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(df_clean["age"], kde=True, ax=axis_grid[0][0])
axis_grid[0][0].set_title("Histogram of age")

sns.boxplot(x=df_clean["age"], ax=axis_grid[0][1])
axis_grid[0][1].set_title("Box plot of age")

sns.histplot(df_clean["fare"], kde=True, ax=axis_grid[1][0])
axis_grid[1][0].set_title("Histogram of fare")

sns.boxplot(x=df_clean["fare"], ax=axis_grid[1][1])
axis_grid[1][1].set_title("Box plot of fare")

figure.tight_layout()
figure.savefig(CHART_DIR + "/task3_univariate_age_fare.png", dpi=120)
plt.close(figure)


def count_outliers_iqr(column_values):
    """
    A simple function that counts how many values are 'outliers'
    using the IQR (Interquartile Range) rule:
      - Q1 = 25th percentile, Q3 = 75th percentile
      - IQR = Q3 - Q1
      - A value is an outlier if it is below (Q1 - 1.5*IQR)
        or above (Q3 + 1.5*IQR)
    """
    q1 = column_values.quantile(0.25)
    q3 = column_values.quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outlier_count = 0
    for value in column_values:
        if value < lower_limit or value > upper_limit:
            outlier_count = outlier_count + 1

    return outlier_count, lower_limit, upper_limit


age_outliers, age_low, age_high = count_outliers_iqr(df_clean["age"])
print("age outliers:", age_outliers, "(normal range:", round(age_low, 2), "to", round(age_high, 2), ")")

fare_outliers, fare_low, fare_high = count_outliers_iqr(df_clean["fare"])
print("fare outliers:", fare_outliers, "(normal range:", round(fare_low, 2), "to", round(fare_high, 2), ")")

# Mean, median and mode of fare, to check if it is skewed
fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]

print("\nfare mean:", round(fare_mean, 2))
print("fare median:", round(fare_median, 2))
print("fare mode:", round(fare_mode, 2))

if fare_mean > fare_median and fare_median > fare_mode:
    print("Conclusion: fare is RIGHT-skewed "
          "(a few very high fares pull the mean up).")
elif fare_mean < fare_median and fare_median < fare_mode:
    print("Conclusion: fare is LEFT-skewed.")
else:
    print("Conclusion: fare is roughly symmetric.")


# =====================================================================
# TASK 4: Compare survival across groups (bivariate analysis)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 4: Survival rate by group")
print("=" * 70)

# (a) Survival rate by sex
print("\nSurvival rate by sex:")
for sex_value in df_clean["sex"].unique():
    rows_for_this_sex = df_clean[df_clean["sex"] == sex_value]
    survival_rate = rows_for_this_sex["survived"].mean()
    print(" ", sex_value, ":", round(survival_rate, 3),
          "(count =", len(rows_for_this_sex), ")")

# (b) Survival rate by passenger class
print("\nSurvival rate by pclass:")
for class_value in sorted(df_clean["pclass"].unique()):
    rows_for_this_class = df_clean[df_clean["pclass"] == class_value]
    survival_rate = rows_for_this_class["survived"].mean()
    print(" class", class_value, ":", round(survival_rate, 3),
          "(count =", len(rows_for_this_class), ")")

# (c) Survival rate by sex AND class together
print("\nSurvival rate by sex + pclass:")
for sex_value in sorted(df_clean["sex"].unique()):
    for class_value in sorted(df_clean["pclass"].unique()):
        rows_here = df_clean[
            (df_clean["sex"] == sex_value) &
            (df_clean["pclass"] == class_value)
        ]
        survival_rate = rows_here["survived"].mean()
        print(" ", sex_value, "+ class", class_value, ":",
              round(survival_rate, 3), "(count =", len(rows_here), ")")

# Correlation matrix on exactly these 6 numeric columns
# (we leave out "adult_male" and "alone" because they are just
# calculated from other columns, not independent measurements)
columns_for_correlation = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
correlation_matrix = df_clean[columns_for_correlation].corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(3))

figure, axis = plt.subplots(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=axis, square=True)
axis.set_title("Correlation heatmap")
figure.tight_layout()
figure.savefig(CHART_DIR + "/task4_correlation_heatmap.png", dpi=120)
plt.close(figure)

# Find the 2 strongest correlations (ignoring the diagonal, which is
# always 1.0 because every column is perfectly correlated with itself)
pair_list = []
for row_name in columns_for_correlation:
    for col_name in columns_for_correlation:
        if row_name != col_name:
            correlation_value = correlation_matrix.loc[row_name, col_name]
            # store pair as a sorted tuple so we don't count
            # (age, fare) and (fare, age) as two separate pairs
            pair = tuple(sorted([row_name, col_name]))
            if pair not in [p[0] for p in pair_list]:
                pair_list.append((pair, correlation_value))

# sort by the size of the correlation (ignoring +/- sign), biggest first
pair_list.sort(key=lambda item: abs(item[1]), reverse=True)

print("\nTop 2 strongest correlations:")
for pair, value in pair_list[:2]:
    print(" ", pair[0], "and", pair[1], "-> correlation =", round(value, 3))


# =====================================================================
# TASK 5: Build a "data story" with at least 4 charts
# =====================================================================
print("\n" + "=" * 70)
print("TASK 5: Data story charts")
print("=" * 70)

# Chart 1: bar chart of survival rate by class and sex
figure, axis = plt.subplots(figsize=(7, 5))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex",
            ax=axis, errorbar=None)
axis.set_title("Survival rate by class and sex")
figure.tight_layout()
figure.savefig(CHART_DIR + "/task5_chart1_bar_class_sex.png", dpi=120)
plt.close(figure)
print("Chart 1: Women in 1st/2nd class survived the most, men in 3rd "
      "class survived the least. Sex matters a lot, and class makes "
      "the gap even bigger.")

# Chart 2: box plot of fare by class and survival
figure, axis = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df_clean, x="pclass", y="fare", hue="survived", ax=axis)
axis.set_title("Fare by class and survival")
figure.tight_layout()
figure.savefig(CHART_DIR + "/task5_chart2_box_fare_class_survival.png", dpi=120)
plt.close(figure)
print("Chart 2: In every class, people who survived usually paid a "
      "bit more for their ticket than people who did not survive.")

# Chart 3: scatter plot of age vs fare, colored by survival
figure, axis = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived",
                 alpha=0.6, ax=axis)
axis.set_title("Age vs fare, colored by survival")
figure.tight_layout()
figure.savefig(CHART_DIR + "/task5_chart3_scatter_age_fare.png", dpi=120)
plt.close(figure)
print("Chart 3: There is no clean line separating survivors from "
      "non-survivors by age and fare alone, but survivors lean "
      "toward higher fares.")

# Chart 4: heatmap of survival rate by class and embarkation port
pivot_table = df_clean.pivot_table(values="survived", index="pclass",
                                    columns="embarked", aggfunc="mean")
figure, axis = plt.subplots(figsize=(6, 5))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", ax=axis)
axis.set_title("Survival rate by class and embarkation port")
figure.tight_layout()
figure.savefig(CHART_DIR + "/task5_chart4_heatmap_class_embarked.png", dpi=120)
plt.close(figure)
print("Chart 4: Survival rate is not the same at every port, even "
      "within the same class - this hints that the port a passenger "
      "boarded from is linked to their wealth/class.")


# =====================================================================
# TASK 6: Standardize age and fare (z-score) as a sanity check
# =====================================================================
print("\n" + "=" * 70)
print("TASK 6: Standardizing age and fare (z-score check)")
print("=" * 70)

print("BEFORE standardizing:")
print("age  -> mean:", round(df_clean["age"].mean(), 3),
      " std:", round(df_clean["age"].std(), 3))
print("fare -> mean:", round(df_clean["fare"].mean(), 3),
      " std:", round(df_clean["fare"].std(), 3))

# z-score formula: (value - mean) / standard_deviation
df_check = df_clean.copy()

age_mean = df_check["age"].mean()
age_std = df_check["age"].std()
df_check["age_z"] = (df_check["age"] - age_mean) / age_std

fare_mean_2 = df_check["fare"].mean()
fare_std_2 = df_check["fare"].std()
df_check["fare_z"] = (df_check["fare"] - fare_mean_2) / fare_std_2

print("\nAFTER standardizing:")
print("age_z  -> mean:", round(df_check["age_z"].mean(), 3),
      " std:", round(df_check["age_z"].std(), 3))
print("fare_z -> mean:", round(df_check["fare_z"].mean(), 3),
      " std:", round(df_check["fare_z"].std(), 3))

print("\nAs expected, after standardizing, mean is about 0 and "
      "standard deviation is about 1. This is just a check - the "
      "actual modeling script (02_modeling.py) does its own "
      "scaling using only the training data.")

figure, axis_grid = plt.subplots(1, 2, figsize=(11, 4))
sns.kdeplot(df_clean["age"], ax=axis_grid[0], label="original")
sns.kdeplot(df_check["age_z"], ax=axis_grid[0], label="standardized")
axis_grid[0].set_title("age: before vs after")
axis_grid[0].legend()

sns.kdeplot(df_clean["fare"], ax=axis_grid[1], label="original")
sns.kdeplot(df_check["fare_z"], ax=axis_grid[1], label="standardized")
axis_grid[1].set_title("fare: before vs after")
axis_grid[1].legend()

figure.tight_layout()
figure.savefig(CHART_DIR + "/task6_standardization_check.png", dpi=120)
plt.close(figure)

print("\nAll done! Charts are saved in the 'charts' folder, and "
      "titanic.csv is saved for the next script (02_modeling.py).")