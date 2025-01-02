import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tf_keras import Sequential
from tf_keras.src.layers import LSTM, Dense

print(tf.__version__)
print(keras.__version__)

# Veriyi yükleme
file_path = "usd_verileri.csv"  # CSV dosyanızın yolu
data = pd.read_csv(file_path, header=None, names=["Date", "Price"])

# Döviz kurundaki virgülleri noktaya çevirme ve float türüne dönüştürme
data["Price"] = data["Price"].astype(str).str.replace(",", ".").astype(float)
# Tarih sütununu datetime formatına çevirme
data["Date"] = pd.to_datetime(data["Date"], format="%d.%m.%Y")

# Tarihi indeks olarak ayarlama
data.set_index("Date", inplace=True)

duplicate_dates = data.index[data.index.duplicated()]
data = data[~data.index.duplicated(keep="first")]

data.resample('10D').mean().plot(figsize=(12,6))

# Veri görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Price"], label="USD/TRY Price", color="blue")
plt.title("USD/TRY Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

n_cols = 1
dataset = data["Price"]
dataset = pd.DataFrame(dataset)
data = dataset.values

scaler = MinMaxScaler(feature_range= (0, 1))
scaled_data = scaler.fit_transform(np.array(data))

# 75% to Train , 25% to Test
train_size = int(len(data) * 0.75)
test_size = len(data) - train_size
print("Train Size :",train_size,"Test Size :",test_size)

# train_data = scaled_data[0:train_size, :]
train_data = scaled_data[-train_size:, :]
train_data.shape

x_train = []
y_train = []
time_steps = 60
n_cols = 1

for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i-time_steps:i, :n_cols])
    y_train.append(train_data[i, :n_cols])
    if i<=time_steps:
        print('X_train: ', x_train)
        print('y_train:' , y_train)

# Convert to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train[::-1]
y_train = y_train[::-1]

x_train.shape , y_train.shape

model = Sequential([
    LSTM(50, return_sequences= True, input_shape= (x_train.shape[1], n_cols)),
    LSTM(64, return_sequences= False),
    Dense(32),
    Dense(16),
    Dense(n_cols)
])

model.compile(optimizer= 'adam', loss= 'mse' , metrics= "mean_absolute_error")

model.summary()

# Fitting the LSTM to the Training set
history = model.fit(x_train, y_train, epochs= 100, batch_size= 32)

plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"])
plt.plot(history.history["mean_absolute_error"])
plt.legend(['Mean Squared Error','Mean Absolute Error'])
plt.title("Losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

print("MSE: ", history.history["loss"][-1])
print("MAE: ", history.history["mean_absolute_error"][-1])

# Creating a testing set with 60 time-steps and 1 output
time_steps = 60
test_data = scaled_data[train_size - time_steps: , :]

x_test = []
y_test = []
n_cols = 1

for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i-time_steps:i, 0:n_cols])
    y_test.append(test_data[i, 0:n_cols])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], n_cols))

x_test.shape , y_test.shape

predictions = model.predict(x_test)

#inverse predictions scaling
predictions = scaler.inverse_transform(predictions)
predictions.shape

y_test = scaler.inverse_transform(y_test)

RMSE = np.sqrt(np.mean( y_test - predictions )**2).round(2)
print("RMSE: ", RMSE)

preds_acts = pd.DataFrame(data={'Predictions':predictions.flatten(), 'Actuals':y_test.flatten()})
preds_acts

plt.figure(figsize = (16, 6))
plt.plot(preds_acts['Predictions'])
plt.plot(preds_acts['Actuals'])
plt.legend(['Predictions', 'Actuals'])
plt.show()

train = dataset.iloc[:train_size , 0:1]
test = dataset.iloc[train_size: , 0:1]
test['Predictions'] = predictions

plt.figure(figsize= (16, 6))
plt.title('Prices Prediction', fontsize= 18)
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Price', fontsize= 18)
plt.plot(train['Price'], linewidth= 3)
plt.plot(test['Price'], linewidth= 3)
plt.plot(test["Predictions"], linewidth= 3)
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()