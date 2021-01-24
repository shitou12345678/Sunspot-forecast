import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\dimensin_8\\m=8t=37.csv')
# sunspot_average_13 = np.array(df)[:, 1]
# t = np.arange(0, 3240)
inv = df.values[:,1]
font = 23
max_index = np.argmax(inv)
min_index = np.argmin(inv)
# plt.figure(1)
# plt.plot(t, sunspot_average_13)
# plt.ylabel('13-month smoothed monthly sunspot number', fontsize=font)
# plt.xlabel('Month', fontsize=font)
# plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], ['1750-04', '1791-09', '1833-05', '1875-01', '1916-09', '1958-05', '2000-01'], fontsize=font)
# plt.show()
plt.plot(inv,'o')
plt.plot(max_index, inv[max_index], 's')
show_max='[2024.4  139.55]'
plt.plot(min_index, inv[min_index], 's')
plt.annotate(show_max, xytext=(max_index, inv[max_index]), xy=(max_index, inv[max_index]),fontsize=font)
plt.xlabel('Month',fontsize=font)
plt.ylabel("Forecast sunspot number of Solar Cycle 25 ",fontsize=font)
plt.xticks([0, 20, 40, 60, 80, 100, 120], ['2020-01', '2021-08', '2023-04', '2024-12', '2026-08', '2028-04', '2029-12'],fontsize=font)
plt.savefig("original.tiff",dpi=300,bbox_inches = 'tight')#解决图片不清晰，不完整的问题