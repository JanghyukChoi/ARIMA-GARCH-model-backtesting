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

# ARIMA 모델 학습
arima_model = ARIMA(train, order=best_order).fit()
arima_resid = arima_model.resid

# GARCH(1,1) 학습
garch_model = arch_model(arima_resid, vol='GARCH', p=1, q=1, mean='Zero')
garch_fit = garch_model.fit(disp='off')

# 테스트 구간에서 예측 및 VaR 계산
forecast_mean = []
forecast_std = []

for i in range(len(test)):
    # 업데이트된 ARIMA 예측
    arima_fit = ARIMA(log_return[:train_size + i], order=best_order).fit()
    mu = arima_fit.forecast(steps=1).iloc[0]
    
    # 업데이트된 GARCH 예측
    resid = arima_fit.resid
    garch_fit = arch_model(resid, vol='GARCH', p=1, q=1, mean='Zero').fit(disp='off')
    sigma = np.sqrt(garch_fit.forecast(horizon=1).variance.values[-1, 0])
    
    forecast_mean.append(mu)
    forecast_std.append(sigma)

forecast_mean = np.array(forecast_mean)
forecast_std = np.array(forecast_std)

# 95% 신뢰수준 일간 VaR
z = norm.ppf(0.05)  # ≈ -1.645
VaR_95 = forecast_mean + z * forecast_std  # 예측 수익률 기준 손실하한

# Exceedance Test
actual_return = test.values
exceedances = actual_return < VaR_95
exceed_rate = np.mean(exceedances)

print(f"총 테스트 일수: {len(test)}")
print(f"VaR를 초과한 횟수: {exceedances.sum()}회")
print(f"실제 초과 비율: {exceed_rate*100:.2f}% (이론: 5%)")

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(test.index, actual_return, label='Actual Return')
plt.plot(test.index, VaR_95, label='VaR (95%)', color='red', linestyle='--')
plt.fill_between(test.index, VaR_95, actual_return, 
                 where=exceedances, color='red', alpha=0.3, label='Exceedance')
plt.title('ARIMA-GARCH 기반 일간 VaR 예측 및 초과 검정')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
