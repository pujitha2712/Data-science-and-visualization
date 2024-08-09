import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Users\achal\OneDrive\Desktop\titanic.csv'

try:
    df = pd.read_csv(file_path)
except PermissionError as e:
    print(f"PermissionError: {e}")
    print("Please check file permissions and try again.")
    quit()

null_columns = df.columns[df.isnull().any()]
print(f"Columns containing null values: {null_columns}")
print(f"Number of columns containing null values: {len(null_columns)}")
print()

print("First 5 rows:")
print(df.head())
print()

print("Last 5 rows:")
print(df.tail())
print()

print("DataFrame info:")
print(df.info())
print()

print("Statistical summary:")
print(df.describe())
print()

print("Shape of the DataFrame:")
print(df.shape)
print()

sns.pairplot(df.dropna())
plt.title("Pairplot of the dataset")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
df_cleaned = df.dropna()

sns.pairplot(df_cleaned)
plt.title("Pairplot of the cleaned dataset")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of the cleaned dataset")
plt.show()
