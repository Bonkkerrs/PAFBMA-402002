import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm # 统计相关的库
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import tushare as ts
import seaborn as sns


#1. data preprocessing
a=0
stocks=pd.read_excel('F:\Python\金工\\tenstocks.xlsx')
print(stocks.index)
columnsname=np.array(stocks.columns)
columnsname1=columnsname.copy()
for i in range(len(columnsname)):  #change name of columns
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
for i in range(10):
    stocks[str(columnsname1[i])+"return"]=stocks[columnsname1[i+10]].pct_change(1)



#2. basic information and ADF test
def tsplot(y, lags=None, title='', figsize=(14, 8)):
    figure = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax
tsplot(stocks["CBT-diff1"], title='', lags=35) #if it does not work you can run following code


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(stocks["CBT-diff1"]  ,use_vlines=True,lags=30)
plt.show()
plot_pacf(stocks["CBT-diff1"]  ,use_vlines=True,lags=30)

#单位根检验
stocks[np.isnan(stocks)]=0
stocks[np.isinf(stocks)]=0
t = adfuller(stocks["CBT-diff1"])
print("p-value:   ",t[1])



#3. difference
stocks["CBT-diff1"] = stocks["CBT"].diff(1).dropna()
stocks["CBT-diff2"] = stocks["CBT-diff1"].diff(1).dropna()
stocks1 = stocks.loc[:,["CBT","CBT-diff1","CBT-diff2"]]
stocks1.plot(subplots=True, figsize=(18, 12),title="差分图")


#4. White noise
 from statsmodels.stats.diagnostic import acorr_ljungbox
 lb=acorr_ljungbox(stocks["AAPL"].diff(1).dropna(), lags = [i for i in range(1,12)],boxpierce=True)
 print("白噪声检验")
 print(lb)




#5. model building and parameters
import itertools

p_min = 0
d_min = 0
q_min = 0
p_max = 4
d_max = 2
q_max = 4

# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                           columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

for p, d, q in itertools.product(range(p_min, p_max + 1),
                                 range(d_min, d_max + 1),
                                 range(q_min, q_max + 1)):
    if p == 0 and d == 0 and q == 0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue

    try:
        model = sm.tsa.ARIMA(stocks["AAPL-diff1"], order=(p, d, q),
                             # enforce_stationarity=False,
                             # enforce_invertibility=False,
                             )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)


#heat map
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')
plt.show()

train_results = sm.tsa.arma_order_select_ic(stocks["CBT-diff1"], ic=['aic', 'bic'], trend='nc', max_ar=10, max_ma=10)
print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)




#6. prediction
stocks.dropna(axis=0, inplace=True)
CBT_fit=np.array(stocks["CBT-diff1"])

for i in range(100):
    arima_CBT=sm.tsa.SARIMAX(CBT_fit[i:1287+i], order=(1,1,1))
    model_results=arima_CBT.fit()
    max_lag = 30
    mdl = smt.ARIMA(CBT_fit[i:1287+i], order=(1,1,1)).fit(
    maxlag=max_lag, method='mle', trend='nc')
    print(mdl.summary())
    forecast_data = model_results.forecast(1)  # 预测未来数据
    CBT_fit=np.append(CBT_fit,forecast_data[0],axis=None)
    print('----------预测未来值')
    print(forecast_data)



plt.plot(stocks["CBT-diff1"], label='原数据')
plt.show()

plt.plot(AAPL_fit, label='未来数据')
plt.show()


# Seasonal data
import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(stocks["CBT-diff1"], model='additive', extrapolate_trend='freq',period=256)
plt.rc('figure',figsize=(12,8))
fig = decomposition.plot()
plt.show()


#7. model test
model_results.plot_diagnostics(figsize=(16,12))

#White test of residual
resid = model_results.resid #赋值
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

res = acorr_ljungbox(resid, lags=24, boxpierce=True, return_df=True)
print(res)
#注：原假设为白噪声（相关系数为零）检验的结果就是看最后一列前十二行的检验概率（一般观察滞后1~12阶），
#如果检验概率小于给定的显著性水平，比如0.05、0.10等就拒绝原假设，即为非白噪声。

#DW test
print('dw的值为')
print(sm.stats.durbin_watson(model_results.resid.values))







