import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from tf_keras import Sequential
from tf_keras.src.layers import LSTM, Dense, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D

sns.set_theme(style="darkgrid", font_scale=1.5)
register_matplotlib_converters()
df = pd.read_csv("usd_verileri.csv", header=None, names=["Date", "Price"], parse_dates=["Date"], dayfirst=True)
df["Price"] = df["Price"].astype(str).str.replace(",", ".").astype(float)

df.set_index("Date", inplace=True)

df.resample('10D').mean().plot(figsize=(12,6))
plt.figtext(.5,0.9,"Down-sampled to 10-day periods", fontsize=20, ha='center')
plt.show()

data = df.iloc[:, 0]

hist = []
target = []
length = 90

for i in range(len(data)-length):
    x = data[i:i+length]
    y = data[i+length]
    hist.append(x)
    target.append(y)

print(len(hist[0]))
print(len(hist))
print(len(target))

print(hist[0][length-1])
print(data[length-1])

print(hist[1][length-1])
print(data[length])
print(target[0])

#convert list to array
hist = np.array(hist)
target = np.array(target)

# Reverse the array
hist = hist[::-1]
target = target[::-1]

target = target.reshape(-1,1)

X_train = hist[:3500,:]
X_test = hist[3500:,:]

y_train = target[:3500]
y_test = target[3500:]

# LSTM Model
# model = Sequential([
#     LSTM(50, return_sequences= True, input_shape= (X_train.shape[1], 1)),
#     LSTM(64, return_sequences= False),
#     Dense(32),
#     Dense(16),
#     Dense(1)
# ])

# GRU Model
# model = Sequential([
#     GRU(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     GRU(64, return_sequences=False),
#     Dense(32, activation='relu'),
#     Dense(16, activation='relu'),
#     Dense(1)
# ])

# Bidirectional LSTM Model
# model = Sequential([
#     Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train.shape[1], 1)),
#     Bidirectional(LSTM(64, return_sequences=False)),
#     Dense(32, activation='relu'),
#     Dense(16, activation='relu'),
#     Dense(1)
# ])

# CNN-LSTM Hybrid Model (BEST)
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Fully Convolutional Networks (FCN) Model
# model = Sequential([
#     Conv1D(filters=128, kernel_size=8, activation='relu', input_shape=(X_train.shape[1], 1)),
#     Conv1D(filters=256, kernel_size=5, activation='relu'),
#     Conv1D(filters=128, kernel_size=3, activation='relu'),
#     GlobalAveragePooling1D(),
#     Dense(50, activation='relu'),
#     Dense(1)
# ])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, epochs=30, batch_size=32)

loss = history.history['loss']
epoch_count = range(1, len(loss) + 1)
plt.figure(figsize=(12,8))
plt.plot(epoch_count, loss, 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

# Prediction

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"MAE: {mae}, RMSE: {rmse}, MSE: {history.history['loss'][-1]}, MAPE: {mae/np.mean(y_test)}, R-Squared: {1 - (np.sum((y_test - pred)**2) / np.sum((y_test - np.mean(y_test))**2))}")

plt.figure(figsize=(12,8))
plt.plot(y_test, color='blue', label='Real')
plt.plot(pred, color='red', label='Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('USD Price Prediction')
plt.legend()
plt.show()
