import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Sample DataFrame for illustration
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

### Visualization Part ###

# Plotting with Seaborn and Matplotlib for historical data + regression line
plt.figure(figsize=(10, 6))

# Scatter plot for historical data
sns.scatterplot(x=df['Date'], y=df['Transaction'], label='Historical Transactions', color='blue')

# Plotting regression line on historical data
plt.plot(df['Date'], model.predict(X), color='red', label='Regression Line')

# Adding predictions for the next 30 days
sns.lineplot(x=future_df['Date'], y=future_df['Predicted_Transaction'], label='Predicted Transactions', color='green', linestyle='--')

plt.title('Historical and Predicted Transactions with Regression Line')
plt.xlabel('Date')
plt.ylabel('Transaction Amount')
plt.legend()
plt.grid(True)
plt.show()

### Plotly Interactive Plot ###

# Combine historical data with predictions for visualization
combined_df = pd.concat([df[['Date', 'Transaction']], future_df[['Date', 'Predicted_Transaction']].rename(columns={"Predicted_Transaction": "Transaction"})])

# Plot interactive plot with Plotly
fig = px.line(combined_df, x='Date', y='Transaction', title='Historical and Predicted Transactions')
fig.add_trace(go.Scatter(x=df['Date'], y=df['Transaction'], mode='markers', name='Historical Transactions'))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Transaction'], mode='lines', name='Predicted Transactions', line=dict(dash='dash')))

fig.update_layout(title='Interactive Historical and Predicted Transactions with Linear Regression',
                  xaxis_title='Date',
                  yaxis_title='Transaction Amount',
                  hovermode="x unified")

fig.show()

### Distribution Visualization (Histogram) ###

# Enhanced Seaborn Histogram with KDE Overlay
plt.figure(figsize=(10, 6))

# Plot histogram with KDE overlay for better distribution visualization
sns.histplot(df['Transaction'], kde=True, bins=15, color='skyblue', edgecolor='black')

# Add statistical lines (mean, median)
mean_val = df['Transaction'].mean()
median_val = df['Transaction'].median()

# Draw vertical lines for the mean and median
plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')

# Customizing plot labels and title
plt.title('Transaction Amount Distribution with KDE', fontsize=16)
plt.xlabel('Transaction Amount', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()

# Show the plot
plt.show()

# Plotly Interactive Histogram with Box Plot for Transaction Distribution
fig = px.histogram(df, x='Transaction', nbins=15, marginal="box", title="Transaction Amount Distribution")
fig.update_traces(marker=dict(color='lightblue'))
fig.update_layout(title='Enhanced Transaction Amount Distribution with Box Plot', xaxis_title='Transaction Amount', yaxis_title='Count')

# Show the interactive plot
fig.show()
