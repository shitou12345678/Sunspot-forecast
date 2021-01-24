import pandas as pd

average_13 = pd.read_csv("C:\\Users\\ZD\\Desktop\\太阳黑子\\模型\\13个月平滑数据集.csv",sep=';', parse_dates={'sunspot_month-13': [0, 1], 'sunspot_number': [3], 'a':[4], 'b':[5],'c':[6],'d':[2]})
average_13 = average_13.drop(['b', 'd', 'c','a'],axis = 1)
average_month = pd.read_csv("C:\\Users\\ZD\\Desktop\\太阳黑子\\模型\\月均值数据集.csv",sep=';', parse_dates={'sunspot_month': [0, 1], 'sunspot_number': [3], 'a':[4], 'b':[5],'c':[6],'d':[2]})
average_month = average_month.drop(['b', 'd', 'c','a'],axis = 1)
average_year = pd.read_csv("C:\\Users\\ZD\\Desktop\\太阳黑子\\模型\\年均值数据集.csv",sep=';', parse_dates={'sunspot_month': [0], 'sunspot_number': [1], 'a':[2], 'b':[3],'c':[4]})
average_year = average_year.drop(['b', 'c','a'],axis = 1)
print(average_13 .head())
print(average_month.head())
print(average_year)
average_13.to_csv('sunspot_average_13.csv', index=False)
average_month.to_csv('sunspot_average.csv',index=False)
average_year.to_csv('average_year.csv',index=False)