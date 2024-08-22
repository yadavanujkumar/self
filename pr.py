import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# Load the actual CSV data
csv_file_path = 'analysis.csv'
df = pd.read_csv(csv_file_path)

# Convert 'TXN Date' to datetime format
df['TXN Date'] = pd.to_datetime(df['TXN Date'], format='%d-%m-%y')

# Fill NaN values in 'DEBIT' and 'CREDIT' columns with 0
df['DEBIT'] = df['DEBIT'].fillna(0)
df['CREDIT'] = df['CREDIT'].fillna(0)

# Calculate net transaction (debit as negative, credit as positive)
df['Transaction'] = df['CREDIT'] - df['DEBIT']

# Sort data by date
df = df.sort_values('TXN Date').reset_index(drop=True)

# Calculate rolling average (7-day window for weekly trend)
df['Rolling_Transaction'] = df['Transaction'].rolling(window=7).mean()

# Plot the rolling average
plt.figure(figsize=(10, 6))
plt.plot(df['TXN Date'], df['Rolling_Transaction'], label='7-Day Rolling Average', color='orange')
plt.scatter(df['TXN Date'], df['Transaction'], label='Original Data', color='blue', alpha=0.5)
plt.title('Transaction Trend with 7-Day Rolling Average')
plt.xlabel('Date')
plt.ylabel('Transaction Amount')
plt.legend()
plt.show()

### 3. Cluster Analysis: Transaction Clustering ###

# Select features for clustering
clustering_features = df[['DEBIT', 'CREDIT', 'BALANCE']].dropna()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clustering_features['Cluster'] = kmeans.fit_predict(clustering_features)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=clustering_features, x='DEBIT', y='CREDIT', hue='Cluster', palette='Set1')
plt.title('Transaction Clustering')
plt.show()

### 9. Advanced Visualization: Heatmaps and Plotly Visualizations ###

# Create a heatmap of transaction frequency by day of the week and hour of the day
df['Hour'] = df['TXN Date'].dt.hour
df['Day of Week'] = df['TXN Date'].dt.dayofweek

heatmap_data = df.groupby(['Day of Week', 'Hour']).size().unstack()

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True)
plt.title('Transaction Frequency by Day of Week and Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.show()

# Plotly visualization for Balance over Time
fig = px.line(df, x='TXN Date', y='BALANCE', title='Balance over Time')
fig.show()

# Plotly scatter plot for Debit vs Credit with Balance
fig = px.scatter(df, x='DEBIT', y='CREDIT', color='BALANCE', title='Debit vs Credit with Balance')
fig.show()

### 11. User Behavior Insights ###

# Calculate spending habits over time
df['Month'] = df['TXN Date'].dt.to_period('M')

# Average monthly spending and income
monthly_summary = df.groupby('Month').agg({
    'DEBIT': 'sum',
    'CREDIT': 'sum',
    'Transaction': 'sum'
}).reset_index()

# Plot spending habits over time
plt.figure(figsize=(10, 6))
plt.plot(monthly_summary['Month'].astype(str), monthly_summary['DEBIT'], label='Total Debit (Spending)', color='red')
plt.plot(monthly_summary['Month'].astype(str), monthly_summary['CREDIT'], label='Total Credit (Income)', color='green')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('Spending and Income Over Time')
plt.legend()
plt.show()

# Analyze typical behavior (e.g., frequency of transactions per month)
behavior_summary = df.groupby('Month').size().reset_index(name='Transaction Count')

plt.figure(figsize=(10, 6))
plt.bar(behavior_summary['Month'].astype(str), behavior_summary['Transaction Count'], color='blue')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.title('Transaction Frequency per Month')
plt.show()
