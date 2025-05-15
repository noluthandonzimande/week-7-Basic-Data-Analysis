import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load Iris dataset from sklearn and convert to pandas DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display first few rows
print("First 5 rows:")
print(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

# No missing values in Iris dataset, but if there were, example to fill:
# df.fillna(method='ffill', inplace=True)
# or drop missing values
# df.dropna(inplace=True)

# Task 2: Basic Data Analysis

# Basic statistics for numerical columns
print("\nBasic statistics:")
print(df.describe())

# Grouping by species and calculating mean of each numerical column
grouped = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped)

# Patterns/Findings:
# - You can observe that different species have distinct average measurements,
#   which is why the Iris dataset is commonly used for classification.

# Task 3: Data Visualization

sns.set(style="whitegrid")

# 1. Line chart - We'll create a mock "time" variable to simulate trends (not real time-series)
df['sample_index'] = range(len(df))
plt.figure(figsize=(10, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset['sample_index'], subset['sepal length (cm)'], label=species)
plt.title('Sepal Length Trend by Species (Sample Index)')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar chart - Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram - Distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot - Sepal length vs Petal length, colored by species
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='bright')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
