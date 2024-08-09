import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
n_samples = 1000
age = np.random.randint(18, 80, size=n_samples)
gender = np.random.choice(['Male', 'Female'], size=n_samples)
impressions = np.random.randint(100, 1000, size=n_samples)
clicks = np.random.randint(10, 200, size=n_samples)
date = np.random.choice(pd.date_range('2023-01-01', periods=365, freq='D'), size=n_samples)
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'impressions': impressions,
    'clicks': clicks,
    'date': date
})
bins = [0, 18, 25, 35, 45, 55, 65, np.inf]
labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
plt.figure(figsize=(12, 6))
sns.violinplot(x='age_group', y='impressions', data=df, inner='quartile')
plt.title('Distribution of Impressions Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Number of Impressions')
plt.show()
df['CTR'] = df['clicks'] / df['impressions']
plt.figure(figsize=(12, 6))
sns.swarmplot(x='age_group', y='CTR', data=df)
plt.title('Distribution of Clickthrough Rate (CTR) Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Clickthrough Rate (CTR)')
plt.show()
plt.figure(figsize=(12, 6))
sns.pointplot(x='age_group', y='CTR', hue='gender', data=df, dodge=True, markers=['o', 's'])
plt.title('CTR Across Age Groups by Gender')
plt.xlabel('Age Group')
plt.ylabel('Clickthrough Rate (CTR)')
plt.legend(title='Gender')
plt.show()
ctr_by_age = df.groupby('age_group')['CTR'].mean()
quantiles_by_age = df.groupby('age_group')['CTR'].quantile([0.25, 0.5, 0.75])
max_by_age = df.groupby('age_group')['CTR'].max()
print("Mean CTR by Age Group:")
print(ctr_by_age)
print("\nQuantiles of CTR by Age Group:")
print(quantiles_by_age)
print("\nMax CTR by Age Group:")
print(max_by_age)
plt.figure(figsize=(12, 6))
ctr_stats = df.groupby(['age_group', 'date'])['CTR'].agg(['mean', 'sem']).reset_index()
sns.lineplot(x='date', y='mean', hue='age_group', data=ctr_stats)
plt.fill_between(ctr_stats['date'], ctr_stats['mean'] - ctr_stats['sem'], ctr_stats['mean'] + ctr_stats['sem'], alpha=0.2)
plt.title('CTR Trends Over Time Across Age Groups')
plt.xlabel('Date')
plt.ylabel('Clickthrough Rate (CTR)')
plt.xticks(rotation=45)
plt.legend(title='Age Group')
plt.tight_layout()
plt.show()