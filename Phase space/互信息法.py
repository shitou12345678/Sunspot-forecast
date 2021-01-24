import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def mutual_information(data, tau_max, n):
    I_sq = np.zeros((tau_max, 1))  # 建立一个空数组，用来存放每个τ值的互信息
    data_length = len(data)  # 计算时间序列的长度
    for tau in range(tau_max):
        S = data[0:data_length-tau]
        Q = data[tau:data_length]
        as1 = np.min(S)
        bq = np.min(Q)
        delts = (np.max(S)-as1)/n
        deltq = (np.max(Q)-bq)/n
        N_sq = np.zeros((n, n))
        for i in range(0, n):
            for j in range(0, n):
                for k in range(0, data_length-tau):
                    as_k = (S[k]-as1)/delts
                    bq_k = (Q[k]-bq)/deltq
                    if as_k >= i-1 and as_k < i and bq_k >=j-1 and bq_k < j:
                        N_sq[i, j] = N_sq[i, j]+1
        Ntotal = np.sum(N_sq)
        Ps = np.sum(N_sq, 0)/Ntotal   # 计算位于一维s格子内的概率
        Pq = np.sum(N_sq.T, 0)/Ntotal  # 计算位于一维q格子内的概率
        Psq = N_sq/Ntotal         # 计算位于二维格子(ii,jj)内概率

        H_s = 0  # 计算S的熵
        H_q = 0  # 计算q的熵
        for i in range(0, n):
            if Ps[i] != 0:
                H_s = H_s - Ps[i] * np.log(Ps[i])
            if Pq[i] != 0:
                H_q = H_q - Pq[i] * np.log(Pq[i])

        H_sq = 0  # 计算(s,q)的联合熵
        for i in range(0, n):
            for j in range(0, n):
                if Psq[i, j] != 0:
                    H_sq = H_sq - Psq[i, j] * np.log(Psq[i, j])

        I_sq[tau] = H_s+H_q-H_sq  # 计算tau下的互信息函数
    return I_sq
df = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\预处理\\sunspot_average_13.csv')
sunspot_average_13 = np.array(df)[:, 1]
I_sq = mutual_information(sunspot_average_13, 60, 100)
fig = plt.figure(1)
plt.plot(I_sq, '*-')
plt.xlabel("Delay time τ")
plt.ylabel("Mutual information T(τ)")
plt.title("Delay time τ by mutual information method")
plt.legend('I(τ)')
plt.grid()
plt.show()