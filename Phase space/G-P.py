import numpy as np
import matplotlib
matplotlib.use('TKAgg')#必须显式指明matplotlib的后端
import matplotlib.pyplot as plt
import pandas as pd
# Backend TkAgg is interactive backend. Turning interactive mode on
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\预处理\\sunspot_average_13.csv')
sunspot_average_13 = np.array(df)[:, 1]  # 时间序列列向量
r = 0.5
log2R = np.linspace(-6, 0, 13)
R = 2**log2R
t = 35
dd = 2
D = np.linspace(2, 30, 15)
p = 50
def CorrelationIntegral(X,tau,M,R,p):
    X = X-np.mean(X)
    X = X/(np.max(X)-np.min(X))
    n = len(X)
    len_m = len(M)
    len_r = len(R)
    Cr = np.zeros((len_m, len_r))
    for u in range(0, len_m):
        for v in range(0, len_r):
            m = M[u]
            r = R[v]
            num = n - (m-1)*tau
            print(num)
            tmp = 0
            for i in range(0, int(num)):
                for j in range(i+p, int(num)):
                    for k in range(1, int(m-1)+1):
                        if np.abs(X[i+k*tau]-X[j+k*tau]) > r:
                            tmp += 1
                            break
            Cr[u, v] = 1 - 2 * tmp/((num - p) * (num - p + 1))
    return Cr

def lm(log2_R, log2_Cr, Linear):
    len_m = np.size(log2_Cr, 1)
    a = np.zeros((len_m, 1))
    b = np.zeros((len_m, 1))
    log2r = np.zeros(len(Linear))
    log2cr = np.zeros(len(Linear))
    Linear.tolist()
    for i in range(0, len_m):
        for j in Linear:
            log2r[Linear.index(j)] = log2_R[int(j)]
            log2cr[Linear.index(j)] = log2_Cr[i, int(j)]
        p = np.polyfit(log2r, log2cr, 1)
        a[i] = p[0]
        b[i] = p[1]
    return [a, b]

correlation = CorrelationIntegral(sunspot_average_13, t, D, R, p)
Log2Cr = np.log2(correlation)
fig1 = plt.figure(1)
plt.plot(log2R, Log2Cr.T, 'k')
plt.xlabel('log2(r)')
plt.ylabel('log2(C(r))')
plt.title("log2(r)-log2(C(r))")
plt.show()
plt.grid
Linear = np.linspace(3, 9, 7)
A, B = lm(log2R, Log2Cr, Linear)
fig2 = plt.figure(2)
plt.plot(D, A, 'k.-')
plt.xlabel('Embedding dimension m')
plt.ylabel('Correlation dimension')
plt.title('Embedding dimension by correlation dimension method')
plt.grid
plt.show()