import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# Load the data
df = pd.read_csv('84618143370889334140_unlocked.csv')  # Example: Load your transaction data from a CSV file

# Example DataFrame structure
data = {
    "Date": ["2024-05-01", "2024-05-05", "2024-05-10", "2024-05-15", "2024-05-20"],
    "Transaction": [-200, 300, -150, 500, -250]  # Negative for withdrawals, positive for deposits
}
df = pd.DataFrame(data)

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort data by date
df = df.sort_values('Date').reset_index(drop=True)

# Extract the days from the first transaction as a feature
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Cleaned Data
print(df.head())

# Preparing the data for linear regression
X = df['Days'].values.reshape(-1, 1)
y = df['Transaction'].values

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Coefficients
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

# Predicting the next 30 days
future_days = np.array(range(df['Days'].max() + 1, df['Days'].max() + 31)).reshape(-1, 1)
predictions = model.predict(future_days)

# Create a DataFrame for future predictions
future_df = pd.DataFrame({
    'Days': future_days.flatten(),
    'Predicted_Transaction': predictions
})

# Convert back to dates for better visualization
future_df['Date'] = df['Date'].max() + pd.to_timedelta(future_df['Days'] - df['Days'].max(), unit='d')

print(future_df.head())

fig = px.scatter(df, x='Date', y='Transaction', title='Historical Transactions')
fig.show()

# Combine historical data with predictions for visualization
combined_df = pd.concat([df[['Date', 'Transaction']], future_df[['Date', 'Predicted_Transaction']].rename(columns={"Predicted_Transaction": "Transaction"})])

# Plot
fig = px.line(combined_df, x='Date', y='Transaction', title='Historical and Predicted Transactions')
fig.add_trace(go.Scatter(x=df['Date'], y=df['Transaction'], mode='markers', name='Historical Transactions'))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Transaction'], mode='lines', name='Predicted Transactions', line=dict(dash='dash')))

fig.show()

# Find the most frequent transaction amount
most_frequent_transaction = df['Transaction'].mode()[0]
print(f"Most frequent transaction amount: {most_frequent_transaction}")

# Visualize the distribution of transactions
fig = px.histogram(df, x='Transaction', title='Transaction Amount Distribution')
fig.show()
