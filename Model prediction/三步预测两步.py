import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from math import sqrt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 将数据转换为监督型数据
def timeseries_to_supervised(data, n_steps_in, n_steps_out):
    X, Y = list(), list()
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        data_x = data[i:end_ix]
        data_y = data[end_ix:out_end_ix]
        X.append(data_x)
        Y.append(data_y)
    X = np.array(X)
    Y = np.array(Y)
    data_x_y = np.concatenate((X, Y), axis=1)
    return data_x_y

df = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\预处理\\sunspot_average_13.csv')
series = df['sunspot_number']
date = df['sunspot_month-13']
date = np.array(date)
series = np.array(series)
series = series.reshape(len(series), 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(series)
n_steps_in = 3
n_steps_out = 2
supervised = timeseries_to_supervised(scaled, n_steps_in, n_steps_out)

# 将第24太阳黑子划分为训练集和测试集
train, test = supervised[0:-132], supervised[-132:]

# 将train_x和train_y分别找出来
train_x = train[:, 0:-2]
train_y = train[:, -2:]
test_x = test[:, 0:-2]
test_y = test[:, -2:]

# 将X，y转换成LSTM的格式输入
n_features = 1
# 输入形式(样本数，时间步长，特征数)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)
train_y = train_y.reshape(train_y.shape[0], train_y.shape[1])
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], n_features)
test_y = test_y.reshape(test_y.shape[0], test_y.shape[1])

# 建立LSTM模型
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_steps_out, activation='relu'))
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_x, train_y, epochs=30, verbose=0, validation_data=(test_x, test_y))
plt.figure(1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# 模型测试训练集
yhat = model.predict(test_x, verbose=0)
yhat = yhat[:, 1]
# 逆缩放
yhat = yhat.reshape(len(yhat), 1)
test_y = test_y[:, 1]
tru = test_y.reshape(len(test_y), 1)
inv_y = scaler.inverse_transform(yhat)
tru_y = scaler.inverse_transform(tru)
rmse = sqrt(mean_squared_error(inv_y, tru_y))
MAE = median_absolute_error(inv_y, tru_y)
MAPE = np.sum(np.abs((inv_y-tru_y)/tru_y))/len(tru_y)*100
print("rmse = ", rmse)
print("MAE = ", MAE)
print("MAPE = ", MAPE)
inv_y1 = pd.DataFrame(inv_y)
inv_y1.to_csv('lstm.csv')
# 画图


# 预测第25周
sunspot25th = np.zeros((66, 2))
num = len(scaled)
for i in range(66):
    pre = scaled[-3:]   # 最后三个数据，预测下两个数据
    pre = pre.reshape(pre.shape[1], pre.shape[0], n_features)  # 用来预测下两步的训练集
    pre_y = model.predict(pre, verbose=0)
    pre_y = pre_y.reshape(pre_y.shape[0]*pre_y.shape[1], 1)
    sunspot25th[i, 0] = pre_y[0]
    sunspot25th[i, 1] = pre_y[1]
    scaled = list(scaled)
    scaled.append(pre_y[0])
    scaled.append(pre_y[1])
    scaled = np.array(scaled)
    scaled = scaled.reshape(len(scaled), 1)
sun = sunspot25th.reshape(len(sunspot25th)*2, 1)
sun_y = scaler.inverse_transform(sun)

font = 23
plt.figure(2)
t = np.arange(0, 660)
plt.plot(t[0:528], series[-528:])
plt.plot(t[396:528], inv_y)
plt.plot(t[528:], sun_y, 'o')
plt.ylabel('13-month smoothed monthly sunspot number', fontsize=font)
plt.xlabel('Month',fontsize=font)
plt.xticks([0, 100, 200, 300, 400, 500, 600], ['1976-01', '1984-04', '1992-08', '2000-12', '2009-04', '2017-08', '2025-12'],fontsize=font)
plt.legend(["Actual",  "Forecast of 24th cycle sunspot", 'Forecast of 25th cycle sunspot'],fontsize=font)
plt.show()