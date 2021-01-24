## LSTM进行单步预测，似乎不太行
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

def scale(train, test):
    # 根据训练数据建立缩放器
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    # 转换train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.fit_transform(train)
    # 转换test data
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
def invert_scale(scaler, value):
    cv = scaler.inverse_transform(value)
    print(cv)
    return cv

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]   # 前面作为训练特征，后面相当于预测值
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    # 添加LSTM层
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))  # 输出层1个node
    # 编译，损失函数mse+优化算法adam
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # 按照batch_size，一次读取batch_size个数据
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# 导入数据
data = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\相空间重构\\X.csv')
data = data.values

# 划分数据集，前90%作为训练集，后10%作为测试集
train, test = train_test_split(data, test_size=0.1)

# 将数据进行归一化
scaler, train_scaled, test_scaled = scale(train, test)

# 构建模型
train_X, train_y = train_scaled[:, 0:-1], train_scaled[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
X_test = test_scaled[:, 0:-1]
Y_test = test_scaled[:, -1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# 拟合网络
history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
fig1 = plt.figure(1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
yhat = model.predict(X_test, batch_size=1)
test_X = X_test.reshape((X_test.shape[0], X_test.shape[2]))
inv_yhat = np.concatenate((yhat, test_X), axis=1)
inv_y = invert_scale(scaler, inv_yhat)
predictions = inv_y[:, 0]
Y_test = Y_test.reshape((len(Y_test),1))
tru_y = np.concatenate((Y_test, test_X), axis=1)
tru_y = invert_scale(scaler, tru_y)
tru_y = tru_y[:, 0]
rmse = sqrt(mean_squared_error(tru_y, predictions))
print('Test RMSE:%.3f' % rmse)
fig2 = plt.figure(2)
plt.plot(tru_y)
plt.plot(predictions)
plt.show()
