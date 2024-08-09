import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import requests
from io import StringIO

url = "https://raw.githubusercontent.com/yijiaceline/Machine-Learning-Zoo-Classification/master/zoo.csv"
response = requests.get(url)
data = response.content.decode('utf-8')
df = pd.read_csv(StringIO(data))
print(df.head())
if 'class_type' not in df.columns:
    raise ValueError("Column 'class_type' not found in the dataset.")
X = df.drop(['animal_name', 'class_type'], axis=1)  # Features
y = df['class_type']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, zero_division=1))
plt.figure(figsize=(12, 8))
sns.pairplot(df.drop(['animal_name'], axis=1), hue='class_type', palette='viridis')
plt.suptitle('Pairplot of Zoo Dataset Features', y=1.02)
plt.show()

corr = df.drop('animal_name', axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='class_type', y='legs', data=df, palette='Set3')
plt.title('Distribution of Legs by Class Type')
plt.xlabel('Class Type')
plt.ylabel('Number of Legs')
plt.show()

plt.figure(figsize=(12, 8))
sns.violinplot(x='class_type', y='legs', data=df, palette='Pastel1')
plt.title('Distribution of Legs by Class Type')
plt.xlabel('Class Type')
plt.ylabel('Number of Legs')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='class_type', data=df, palette='viridis')
plt.title('Distribution of Class Types')
plt.xlabel('Class Type')
plt.ylabel('Count')
plt.show()
