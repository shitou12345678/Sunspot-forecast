import pandas as pd
import matplotlib.pyplot as plt
psr_lstm = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\太阳黑子LSTM\\psr-lstm.csv')
psr_bp = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\太阳黑子LSTM\\psr-bp.csv')
psr_tcn = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\太阳黑子LSTM\\psr-tcn.csv')
naive_lstm = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\太阳黑子LSTM\\lstm.csv')
tru_y = pd.read_csv('C:\\Users\\ZD\\PycharmProjects\\太阳黑子\\太阳黑子LSTM\\4.csv')
lstm = psr_lstm.values[:,1]
bp = psr_bp.values[:,1]
tcn = psr_tcn.values[:,1]
naive_lstm = naive_lstm.values[:,1]
tru = tru_y.values[:,1]
obso_psrlstm = tru - lstm
obso_lstm = tru - naive_lstm
obso_bp = tru - bp
obso_tcn = tru - tcn
comp_psrlstm = obso_psrlstm/tru
comp_lstm = obso_lstm/tru
comp_bp = obso_bp/tru
comp_tcn = obso_tcn/tru
font=23
plt.figure()
plt.subplot(2,1,1)
plt.plot(lstm)
plt.plot(bp)
plt.plot(naive_lstm)
plt.plot(tcn)
plt.plot(tru)
plt.ylabel('sunspot number', fontsize=font)
# plt.xlabel('Month', fontsize=font)
plt.legend(['PSR-LSTM','PSR-BP','LSTM','PSR-TCN','Ground Truth'])
plt.xticks([0, 20, 40, 60, 80, 100, 120], ['2009-01', '2010-09', '2012-05', '2013-12', '2015-08', '2017-04', '2018-12'],fontsize=font)
plt.subplot(2,1,2)
plt.plot(obso_bp,'o-')
plt.plot(obso_lstm,'o-')
plt.plot(obso_psrlstm,'o-')
plt.plot(obso_tcn,'o-')
plt.ylabel('Absolute error', fontsize=font)
plt.legend(['PSR-BP','LSTM','PSR-LSTM','PSR-TCN'])
plt.xlabel('Month', fontsize=font)
plt.xticks([0, 20, 40, 60, 80, 100, 120], ['2009-01', '2010-09', '2012-05', '2013-12', '2015-08', '2017-04', '2018-12'],fontsize=font)
# plt.xlabel('Month', fontsize=font)
# plt.subplot(3,1,3)
# plt.plot(comp_bp,'o-')
# plt.plot(comp_lstm,'o-')
# plt.plot(comp_psrlstm,'o-')
# plt.plot(comp_tcn,'o-')
# plt.legend(['PSR-BP','LSTM','PSR-LSTM','PSR-TCN'])
# plt.ylabel('Relative error', fontsize=font)
# # plt.xticks([0, 20, 40, 60, 80, 100, 120], ['2009-01', '2010-09', '2012-05', '2013-12', '2015-08', '2017-04', '2018-12'],fontsize=font)
# plt.show()