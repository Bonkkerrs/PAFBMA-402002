import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


#原数据的简单处理和导入
a=0
stocks=pd.read_excel('F:\Python\金工\\tenstocks.xlsx') #读取数据df.head()
print(stocks.index)
columnsname=np.array(stocks.columns)
columnsname1=columnsname.copy()
for i in range(len(columnsname)):  #修改列名
    if "." in columnsname[i]:
        if "1" in columnsname[i]:
            columnsname1[i]=columnsname[i].replace(".1","-close")
        elif "2" in columnsname[i]:
            columnsname1[i]=columnsname[i].replace(".2","-high")
        elif "3" in columnsname[i]:
            columnsname1[i]=columnsname[i].replace(".3","-low")
        elif "4" in columnsname[i]:
            columnsname1[i]=columnsname[i].replace(".4","-open")
        elif "5" in columnsname[i]:
            columnsname1[i]=columnsname[i].replace(".5","-volume")
    stocks=stocks.rename({stocks.columns[i]:columnsname1[i]}, axis=1)
    print(stocks['Date'])  # 显示导入的数据

drop=[]
for i in range(len(columnsname1)):
    if 'TSLA' in columnsname1[i]:
        continue
    else:
        drop.append(columnsname1[i])
s = stocks.drop(drop, axis=1)
s = pd.DataFrame(s)
s = s[251:351]
s = s.dropna(axis=1)

cor = s.iloc[:,0:6].corr(method='pearson')
print(cor)
#相关系数热力图
sns.heatmap(cor,
            annot=True,  # 显示相关系数的数据
            center=0.5,  # 居中
            fmt='.2f',  # 只显示两位小数
            linewidth=0.5,  # 设置每个单元格的距离
            linecolor='blue',  # 设置间距线的颜色
            vmin=0, vmax=1,  # 设置数值最小值和最大值
            xticklabels=True, yticklabels=True,  # 显示x轴和y轴
            square=True,  # 每个方格都是正方形
            cbar=True,  # 绘制颜色条
            cmap='Blues',  # 设置热力图颜色
            )
plt.show() #显示图片



x_train=0
y_train=0
x_test=0
y_test=0
stockname1=0
def dataset(stockname):
    global x_train, y_train, x_test ,y_test,x_train_return1,y_train_return1,x_test_return1,y_test_return1
    drop=[]
    for i in range(len(columnsname1)):
        if stockname in columnsname1[i]:
            continue
        else:
            drop.append(columnsname1[i])
    stockname1 = stocks.drop(drop, axis=1)
    stockname1 = pd.DataFrame(stockname1)
    stockname1 = stockname1[251:351]
    stockname1 = stockname1.dropna(axis=1)
    values = stockname1.values
    scaler = MinMaxScaler(feature_range=(0, 1)) #归一化
    scaled_stock = scaler.fit_transform(values.reshape(-1, 6))

    train_data_return,test_data_return= train_test_split(stockname1.iloc[:,:6], test_size = 0.3)
    x_train_return1=train_data_return.iloc[:,1:5].copy()
    y_train_return1=train_data_return.iloc[:,0].copy()
    x_test_return1=test_data_return.iloc[:,1:5].copy()
    y_test_return1=test_data_return.iloc[:,0].copy()



    training_data_len = math.ceil(len(values) * 0.7) #股价训练集
    train_data = scaled_stock[0: training_data_len, :]
    train_data = pd.DataFrame(train_data).copy()
    x_train = train_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]

    test_data = scaled_stock[training_data_len:, :]#股价训练集
    test_data = pd.DataFrame(test_data).copy()
    x_test = test_data.iloc[:, 1:]
    y_test = test_data.iloc[:, 0]

    return x_train, y_train ,x_test,y_test,x_train_return1,y_train_return1,x_test_return1,y_test_return1



def model_price(x_train,y_train,x_test,y_test):
    # 股价训练集测试结果
    clf_train = SVR(kernel='linear')
    clf_train.fit(x_train, y_train)
    y_hat_train = clf_train.predict(x_train)
    print("得分:", r2_score(y_train, y_hat_train))
    print(y_test.shape, y_hat_train.shape, x_test.shape)

    # 测试集测试结果
    clf = SVR(kernel="linear", gamma=0.005,
              coef0=0.0, tol=0.00001, C=1.0, epsilon=0.001,
              shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    print("R2:", r2_score(y_test, y_hat))
    print(y_test.shape, y_hat.shape, x_test.shape)

    # 股价训练集测试结果均方根误差
    def sem_rsem(Predict, Original):
        from sklearn.metrics import mean_squared_error
        sem = mean_squared_error(Original, Predict)
        rmse = np.sqrt(sem)
        return sem, rmse

    print(sem_rsem(Predict=y_hat_train, Original=y_train))

    # 股价测试集测试结果均方根误差
    def sem_rsem(Predict, Original):
        from sklearn.metrics import mean_squared_error
        sem = mean_squared_error(Original, Predict)
        rmse = np.sqrt(sem)
        return sem, rmse

    print(sem_rsem(Predict=y_hat, Original=y_test))

    # 股价结果可视化
    r = len(x_test) + 1
    print(y_test)
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, r), y_hat, 'go-', label="predict")
    plt.plot(np.arange(1, r), y_test, 'co-', label="real")
    plt.legend()
    plt.show()






dataset('AAPL')
model_price(x_train,y_train,x_test,y_test)

dataset('CBT')
model_price(x_train,y_train,x_test,y_test)

dataset('EQIX')
model_price(x_train,y_train,x_test,y_test)

dataset('GS')
model_price(x_train,y_train,x_test,y_test)

dataset('NFLX')
model_price(x_train,y_train,x_test,y_test)

dataset('PFE')
model_price(x_train,y_train,x_test,y_test)

dataset('SHEL')
model_price(x_train,y_train,x_test,y_test)

dataset('TSLA')
model_price(x_train,y_train,x_test,y_test)

dataset('UPS')
model_price(x_train,y_train,x_test,y_test)

dataset('MHT')
model_price(x_train,y_train,x_test,y_test)







