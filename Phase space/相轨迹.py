import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
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
Xn, Yn, X = PhaSpaRecon(sunspot_average_13, tau=37, d=7, T=1)
X = X.values
fig = plt.figure()
# fig.set_facecolor('blueviolet')
ax1 = plt.axes(projection='3d')
z = X[:3000, 1]
x= sunspot_average_13[:3000]
y = X[:3000, 0]
# zd = 13*np.random.random(100)
# xd = 5*np.sin(zd)
# yd = 5*np.cos(zd)
# ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
ax1.plot3D(x, y, z)    #绘制空间曲线
ax1.set_xlabel('s(t)')
ax1.set_ylabel('s(t+τ)')
ax1.set_zlabel('s(t+2τ)')
ax1.set_title('Phase Space Reconstruction')
plt.show()
