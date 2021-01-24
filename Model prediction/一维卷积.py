import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling1D
from keras.layers import Convolution1D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from keras.layers import LSTM
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
        Xn1 = np.zeros((d, lens-(d-1)*tau-1))
        for i in range(0, d):
            Xn1[i, :] = data[i*tau:i*tau+lens-(d-1)*tau-1, 0]
            # for j in range(i*tau, i*tau+lens-(d-1)*tau-1):
            #     Xn[i, j] = data[j]
        Yn1 = data[(T+(d-1)*tau):T+(d-1)*tau+lens-(d-1)*tau-1, 0]
        Yn = Yn1.reshape((len(Yn1), 1))
        Yn = pd.DataFrame(Yn)
        Xn = Xn1.reshape((Xn1.shape[1], Xn1.shape[0]))
        Xn = pd.DataFrame(Xn)
        X = pd.concat([Xn, Yn], axis=1)
    return Xn, Yn,  X
# Xn, Yn = PhaSpaRecon(sunspot_average_13, tau=37, d=7, T=1)  # 一步预测的训练集

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

n_features = 1
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)
# train_y = train_y.reshape(train_y.shape[0], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], n_features)
# test_y = test_y.reshape(test_y.shape[0], 1)

model = Sequential()
model.add(Dense(128, input_shape=(7, 1)))
model.add(Convolution1D(filters=32, kernel_size=3, padding='same'))
# 激活层1
model.add(Activation('relu'))
# 池化层1
model.add(MaxPooling1D(pool_size=2))
# 卷积层2
model.add(Convolution1D(filters=32, kernel_size=3, padding='same'))
# 激活层2
model.add(Activation('relu'))
# 池化层2
model.add(MaxPooling1D(pool_size=2))
# 卷积层3
model.add(Convolution1D(filters=32, kernel_size=3, padding='same'))
# 激活层3
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
# 卷积层4
model.add(Convolution1D(filters=32, kernel_size=3, padding='same'))
# 激活层4
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
# 拉成一维数据
model.add(Flatten())
# 全连接层1
model.add(Dense(512), activation='relu')
# 激活层
model.add(Activation('relu'))
# 全连接层2
model.add(Dense(256), activation='relu')
# 激活层
model.add(Activation('relu'))
# 随机失活
model.add(Dropout(0.25))
# 全连接层2
model.add(Dense(1), activation='relu')
print(model.summary())

model.compile(optimizer='adam', loss='mse')
history = model.fit(train_x, train_y, epochs=100, verbose=0, validation_data=(test_x, test_y))
plt.figure(1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat = model.predict(test_x, verbose=0)
inv_y = scal.inverse_transform(yhat)
test_y = test_y.reshape(test_y.shape[0], 1)
tru_y = scal.inverse_transform(test_y)
RMSE = sqrt(mean_squared_error(tru_y, inv_y))
MAE = median_absolute_error(tru_y, test_y)
MAPE = np.sum(np.abs((inv_y-tru_y)/tru_y))/len(tru_y) * 100
print("rmse = ", RMSE)
print("MAE = ", MAE)
print("MAPE = ", MAPE)
