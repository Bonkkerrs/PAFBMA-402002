import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statistics
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
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


# x_train=0
# y_train=0
# x_test=0
# y_test=0


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
    scaled_stock = values.reshape(-1, 6)

    training_data_len = math.ceil(len(values) * 0.7)  # 股价训练集
    train_data = scaled_stock[0: training_data_len, :]
    train_data = pd.DataFrame(train_data).copy()
    x_train = train_data.iloc[:, 1:5]
    y_train = train_data.iloc[:, 0]

    test_data = scaled_stock[training_data_len:, :]  # 股价训练集
    test_data = pd.DataFrame(test_data).copy()
    x_test = test_data.iloc[:, 1:5]
    y_test = test_data.iloc[:, 0]


    # train_data_return,test_data_return= train_test_split(scaled_stock.iloc[:,:5], test_size = 0.3)
    # x_train_return1=train_data_return.iloc[:,1:4].copy()
    # y_train_return1=train_data_return.iloc[:,0].copy()
    # x_test_return1=test_data_return.iloc[:,1:4].copy()
    # y_test_return1=test_data_return.iloc[:,0].copy()

    return x_train,y_train,x_test,y_test




def model_return(x_train, y_train, x_test, y_test):
    x_train_return = x_train.pct_change(1)
    y_train_return = y_train.pct_change(1)
    x_test_return = x_test.pct_change(1)
    y_test_return = y_test.pct_change(1)

    for i in range(x_train_return.shape[1]):
        temp_col = np.array(x_train_return)[:, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        temp_col = x_train_return.iloc[:, i]
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()

        temp_col = np.array(y_train_return)
        nan_num = np.count_nonzero(temp_col != temp_col)
        temp_col = y_train_return
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()

    for i in range(x_test_return.shape[1]):
        temp_col = np.array(x_test_return)[:, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        temp_col = x_test_return.iloc[:, i]
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()


        temp_col = np.array(y_test_return)
        nan_num = np.count_nonzero(temp_col != temp_col)
        temp_col = y_test_return
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()




    # 训练集测试结果
    clf_train = SVR(kernel='rbf', degree=3, gamma=0.1,
                    coef0=0.0, tol=0.001, C=1.0, epsilon=0.1,
                    shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
    clf_train.fit(x_train_return, y_train_return)
    y_hat_train = clf_train.predict(x_train_return)
    print("得分:", r2_score(y_train_return, y_hat_train))

    # 测试集测试结果
    clf = SVR(kernel="linear", gamma=0.005,
              coef0=0.0, tol=0.00001, C=1.0, epsilon=0.001,
              shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
    clf.fit(x_train_return, y_train_return)
    y_hat = clf.predict(x_test_return)
    print("R2:", r2_score(y_test_return, y_hat))



    # 测试集测试结果均方根误差
    def sem_rsem(Predict, Original):
        from sklearn.metrics import mean_squared_error
        sem = mean_squared_error(Original, Predict)
        rmse = np.sqrt(sem)
        return sem, rmse

    print(sem_rsem(Predict=y_hat, Original=y_test))

    #平均收益
    mean_return=statistics.mean(y_hat)
    print(mean_return)




    # 可视化
    r = len(x_test_return) + 1
    print(y_test_return)
    plt.plot(np.arange(1, r), y_test_return, 'go-', label="predict")
    plt.plot(np.arange(1, r), y_hat, 'co-', label="real")
    plt.legend()
    plt.show()


dataset('AAPL')
model_return(x_train,y_train,x_test,y_test)

dataset('CBT')
model_return(x_train,y_train,x_test,y_test)

dataset('EQIX')
model_return(x_train,y_train,x_test,y_test)

dataset('GS')
model_return(x_train,y_train,x_test,y_test)

dataset('NFLX')
model_return(x_train,y_train,x_test,y_test)

dataset('PFE')
model_return(x_train,y_train,x_test,y_test)

dataset('SHEL')
model_return(x_train,y_train,x_test,y_test)

dataset('TSLA')
model_return(x_train,y_train,x_test,y_test)

dataset('UPS')
model_return(x_train,y_train,x_test,y_test)

dataset('WMT')
model_return(x_train,y_train,x_test,y_test)
