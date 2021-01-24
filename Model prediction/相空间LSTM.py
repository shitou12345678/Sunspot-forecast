# 似乎不太行
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def PhaSpaRecon(df, tau, d, T):
    data = df
    lens = len(data)
    if (lens - T - (d-1) * tau) < 1:
        print("error: delay time or the embedding dimension is too large")
    else:
        Xn1 = np.zeros((lens-(d-1)*tau-1, d))
        Yn1 = [0] * (lens-(d-1)*tau-1)
        for i in range(0, lens-(d-1)*tau-1):
            for j in range(d):
                Xn1[i, j] = data[i + j * tau]

            Yn1[i] = data[i+1 + (d-1) * tau]
        Yn1 = np.array(Yn1)
        Yn = Yn1.reshape((len(Yn1), 1))
        Yn = pd.DataFrame(Yn)
        Xn = pd.DataFrame(Xn1)
        X = pd.concat([Xn, Yn], axis=1)
    return Xn, Yn,  X


df = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\预处理\\sunspot_average_13.csv')
sunspot_average_13 = np.array(df)[:, 1]
# data为时间序列，tau为重构的时延，d为重构的维数，T为直接预测的步数
scal = MinMaxScaler(feature_range=(0, 1))
dat = sunspot_average_13.reshape(len(sunspot_average_13), 1)
scaled = scal.fit_transform(dat)
Xn, Yn, X = PhaSpaRecon(scaled, tau=37, d=7, T=1)    # 这里Xn和Yn是监督型学习的特征和标签吧,相当于有7个变量预测下一步
train = X.values
train_x = train[:-132, 0:7]
train_y = train[:-132, 7]
test_x = train[-132:, 0:7]
test_y = train[-132:, 7]

# 建立输入的模型
n_features = 1
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)
train_y = train_y.reshape(train_y.shape[0], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], n_features)
test_y = test_y.reshape(test_y.shape[0], 1)

# 建立模型
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(7, 1)))
model.add(LSTM(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mse')
print(model.summary())
history = model.fit(train_x, train_y, epochs=30, verbose=0)
# plt.figure(1)
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='validation')
# plt.legend()
# plt.show()

# 预测24
yhat = model.predict(test_x, verbose=0)
inv_y = scal.inverse_transform(yhat)
tru_y = scal.inverse_transform(test_y)
RMSE = sqrt(mean_squared_error(tru_y, inv_y))
MAE = median_absolute_error(tru_y, test_y)
MAPE = np.sum(np.abs((inv_y-tru_y)/tru_y))/len(tru_y) * 100
print("rmse = ", RMSE)
print("MAE = ", MAE)
print("MAPE = ", MAPE)

plt.figure(1)
plt.plot(inv_y, 'b.-')
plt.plot(tru_y, 'r.-')
plt.legend(['预测值', '真实值'])
plt.show()
inv_y1 = pd.DataFrame(inv_y)
inv_y1.to_csv('psr-lstm.csv')
# 预测第25周

tau = 37
m = tau * 6 + 1
sunspot25th = [0] * 132
num = len(scaled)
for i in range(132):
    data_25th = scaled[-m:]
    sun = np.zeros((1, 7))
    for j in range(7):
        sun[0, j] = data_25th[j * tau]
    sun = sun.reshape(1, 7, n_features)
    pre_y = model.predict(sun, verbose=0)
    pre25th = scal.inverse_transform(pre_y)
    sunspot25th[i] = pre25th[0][0]
    scaled = list(scaled)
    scaled.append(pre_y[0])
    scaled = np.array(scaled)
    scaled = scaled.reshape(len(scaled), 1)

font = 23
plt.figure(2)
t = np.arange(0, 660)
plt.plot(t[0:528], sunspot_average_13[-528:])
plt.plot(t[396:528], inv_y)
plt.plot(t[528:], sunspot25th, 'o')
plt.ylabel('13-month smoothed monthly sunspot number', fontsize=font)
plt.xlabel('Month', fontsize=font)
plt.xticks([0, 100, 200, 300, 400, 500, 600], ['1976-01', '1984-04', '1992-08', '2000-12', '2009-04', '2017-08', '2025-12'],fontsize=font)
plt.legend(["Actual",  "Forecast of 24th cycle sunspot", 'Forecast of 25th cycle sunspot'],fontsize=font)
plt.show()
