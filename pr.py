import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import calendar

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

### Linear Regression and Future Prediction ###

# Prepare data for linear regression
df['Days'] = (df['TXN Date'] - df['TXN Date'].min()).dt.days
X = df['Days'].values.reshape(-1, 1)
y = df['Transaction'].values

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict future transactions
last_date = df['TXN Date'].max()
days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
days_left = days_in_month - last_date.day

future_days = np.array(range(df['Days'].max() + 1, df['Days'].max() + days_left + 1)).reshape(-1, 1)
predictions = model.predict(future_days)

# Create a DataFrame for future predictions
future_df = pd.DataFrame({
    'Days': future_days.flatten(),
    'Predicted_Transaction': predictions
})

# Convert future days back to dates
future_df['Date'] = last_date + pd.to_timedelta(future_df['Days'] - df['Days'].max(), unit='d')

# Visualization: Historical and Predicted Transactions with Linear Regression
plt.figure(figsize=(12, 7))
plt.scatter(df['TXN Date'], df['Transaction'], label='Historical Transactions', color='blue', alpha=0.5)
plt.plot(df['TXN Date'], model.predict(X), color='red', label='Regression Line')
plt.plot(future_df['Date'], future_df['Predicted_Transaction'], color='green', linestyle='--', label='Predicted Transactions')

# Adjust y-axis limits
plt.ylim(min(y.min(), predictions.min()) - 10, max(y.max(), predictions.max()) + 10)

plt.title('Historical and Predicted Transactions with Linear Regression')
plt.xlabel('Date')
plt.ylabel('Transaction Amount')
plt.legend()
plt.grid(True)
plt.show()

# Plotly Interactive Plot for Transactions
combined_df = pd.concat([df[['TXN Date', 'Transaction']], future_df[['Date', 'Predicted_Transaction']].rename(columns={"Predicted_Transaction": "Transaction"})])

fig = px.line(combined_df, x='Date', y='Transaction', title='Historical and Predicted Transactions')
fig.add_trace(go.Scatter(x=df['TXN Date'], y=df['Transaction'], mode='markers', name='Historical Transactions'))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Transaction'], mode='lines', name='Predicted Transactions', line=dict(dash='dash')))

# Adjust y-axis range
fig.update_yaxes(range=[min(combined_df['Transaction'].min(), predictions.min()) - 10, max(combined_df['Transaction'].max(), predictions.max()) + 10])

fig.update_layout(title='Interactive Historical and Predicted Transactions with Linear Regression', xaxis_title='Date', yaxis_title='Transaction Amount', hovermode="x unified")
fig.show()

### Transaction Behavior Analysis ###

# Calculate transaction counts and summary statistics by month
df['Month'] = df['TXN Date'].dt.to_period('M')
monthly_summary = df.groupby('Month').agg({
    'DEBIT': 'sum',
    'CREDIT': 'sum',
    'Transaction': 'sum'
}).reset_index()

# Plot spending and income over time
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

### Transactional Patterns and Frequency Analysis ###

# Daily Transaction Patterns
df['Day of Week'] = df['TXN Date'].dt.day_name()
daily_pattern = df.groupby('Day of Week').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.figure(figsize=(10, 6))
daily_pattern.plot(kind='bar', color='purple')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Transactions')
plt.title('Transaction Frequency by Day of the Week')
plt.show()

# Most and Least Frequent Transactions
transaction_counts = df['Transaction'].value_counts()

most_frequent = transaction_counts.idxmax()
least_frequent = transaction_counts.idxmin()
most_frequent_count = transaction_counts.max()
least_frequent_count = transaction_counts.min()

print(f"Most Frequent Transaction Amount: {most_frequent} (Count: {most_frequent_count})")
print(f"Least Frequent Transaction Amount: {least_frequent} (Count: {least_frequent_count})")

# Plot the distribution of transaction amounts
plt.figure(figsize=(12, 6))
sns.histplot(df['Transaction'], bins=30, kde=True, color='teal')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Amounts')
plt.show()
