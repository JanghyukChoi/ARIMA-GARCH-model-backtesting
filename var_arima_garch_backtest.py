import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from itertools import product
from scipy.stats import norm

# 데이터 로딩
aapl = fdr.DataReader('AAPL', '2018-01-01', '2025-06-12')
log_return = np.log(aapl['Close']).diff().dropna()

# 훈련/테스트 분리
train_size = int(len(log_return) * 0.8)
train, test = log_return[:train_size], log_return[train_size:]

# ARIMA 최적 모수 찾기
best_aic = np.inf
best_order = None
for p, q in product(range(0, 4), range(0, 4)):
    try:
        model = ARIMA(train, order=(p, 0, q)).fit()
        if model.aic < best_aic:
            best_aic = model.aic
            best_order = (p, 0, q)
    except:
        continue

# 백테스트 시작
z = norm.ppf(0.05)  # 95% VaR
capital = 1.0  # 초기 투자금
positions = []  # 포지션 기록
returns = []  # 일별 수익률 기록
signal = 'HOLD'
holding = False
exceedance_record = []

for i in range(len(test)):
    history = log_return[:train_size + i]
    
    arima_fit = ARIMA(history, order=best_order).fit()
    mu = arima_fit.forecast(steps=1).iloc[0]
    
    resid = arima_fit.resid
    garch_fit = arch_model(resid, vol='GARCH', p=1, q=1, mean='Zero').fit(disp='off')
    sigma = np.sqrt(garch_fit.forecast(horizon=1).variance.values[-1, 0])
    
    var_95 = mu + z * sigma
    actual_return = test.iloc[i]
    
    # exceed 기록
    exceed = actual_return < var_95
    exceedance_record.append(exceed)
    
    # 최근 2일 exceed 체크
    recent_exceeds = exceedance_record[-2:]
    exceed_risk = sum(recent_exceeds) >= 2 if len(recent_exceeds) == 2 else False
    
    # 전략 조건에 따른 시그널 판단
    if mu > 0 and var_95 > -0.03:
        signal = 'LONG'
    if mu < 0 or sigma > 0.03 or exceed_risk:
        signal = 'EXIT'
    
    # 수익률 계산
    if signal == 'LONG':
        returns.append(actual_return)
        holding = True
    else:
        returns.append(0)
        holding = False

    positions.append(signal)

# 결과 분석
strategy_return = np.cumsum(returns)
buy_and_hold_return = np.cumsum(test.values)

import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.plot(test.index, strategy_return, label='Strategy Cumulative Return')
plt.plot(test.index, buy_and_hold_return, label='Buy & Hold Cumulative Return', linestyle='--')
plt.title('ARIMA-GARCH Based Backtest Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 성과 지표
total_return = strategy_return[-1]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
max_drawdown = np.max(np.maximum.accumulate(strategy_return) - strategy_return)

total_return, sharpe_ratio, max_drawdown