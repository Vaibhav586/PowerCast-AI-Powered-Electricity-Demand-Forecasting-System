import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv(r"C:\Users\vikas\OneDrive\Documents\SIH'24\dataset_tk.csv")

# Extract necessary columns and process datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d-%m-%Y %H.%M", errors='coerce')
df = df[['Datetime', 'Delhi']]  # Only using the 'Delhi' energy consumption column

# Drop any rows with NaN values
df.dropna(subset=['Datetime', 'Delhi'], inplace=True)

# Feature Engineering
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day

# Plot to visualize the data
sns.lineplot(x=df['Datetime'], y=df['Delhi'])
plt.title("Delhi Energy Consumption Over Time")
plt.xticks(rotation=90)
plt.show()

# Preparing the dataset for training
new_data = df.set_index('Datetime')['Delhi']

# Splitting into training and test sets
train_size = int(len(new_data) * 0.8)
train_data = new_data[:train_size]
test_data = new_data[train_size - 60:]  # Keeping last 60 points of train for test sequence

# Normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))

# Creating sequences for LSTM model
def create_sequences(data, window_size=60):
    X, Y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

X_train, Y_train = create_sequences(train_scaled)

# Reshape for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the enhanced LSTM model
model = Sequential()

# Layer 1: Bidirectional LSTM
model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1))))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Layer 2: Stacked LSTM
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Layer 3: GRU Layer for diversity in learning patterns
model.add(GRU(units=50, return_sequences=False))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Output layer
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

# Testing phase
test_scaled = scaler.transform(test_data.values.reshape(-1, 1))
X_test, Y_test = create_sequences(test_scaled)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_values = model.predict(X_test)
predicted_values = scaler.inverse_transform(predicted_values)

# Plotting predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(test_data.index[60:], test_data.values[60:], label="Actual Data")
plt.plot(test_data.index[60:], predicted_values, label="Predicted Data")
plt.legend()
plt.title("Delhi Energy Consumption Prediction")
plt.show()

# Function to predict future consumption for a given date
def predict_for_date(future_date_str, model, last_sequence, num_days=60):
    """
    Predict energy consumption for a future date based on previous data.

    Args:
    - future_date_str: Date in 'YYYY-MM-DD' format for which prediction is needed.
    - model: Trained LSTM model.
    - last_sequence: The last 60 days of data used as input.
    - num_days: Number of future steps to predict.
    """
    future_date = pd.to_datetime(future_date_str)
    current_sequence = last_sequence.reshape(1, num_days, 1)
    
    # Predict the next value
    future_prediction = model.predict(current_sequence)
    
    # Inverse transform the predicted value
    future_consumption = scaler.inverse_transform(future_prediction)[0][0]
    return future_consumption

# Example prediction for future date (e.g., 2021-01-01)
last_60_days_data = test_scaled[-60:]
future_date = "2021-01-01"
predicted_value = predict_for_date(future_date, model, last_60_days_data)
print(f"Predicted energy consumption for {future_date}: {predicted_value} MW")
