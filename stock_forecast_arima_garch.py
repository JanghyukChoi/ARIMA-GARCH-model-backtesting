import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import FinanceDataReader as fdr

# 1. 데이터 수집
aapl = fdr.DataReader('AAPL', '2018-01-01', '2025-06-12')
aapl_close = aapl['Close'].dropna()

# 2. 로그 수익률 계산
log_price = np.log(aapl_close)
log_return = log_price.diff().dropna()

# 3. ARIMA 모형 탐색 (Grid Search)
p_range, q_range = range(0, 5), range(0, 5)
results = []

for p, q in product(p_range, q_range):
    try:
        model = ARIMA(log_return, order=(p, 0, q))
        result = model.fit()
        results.append({'order': (p, 0, q), 'aic': result.aic, 'bic': result.bic})
    except:
        continue

results_df = pd.DataFrame(results).sort_values('aic').reset_index(drop=True)
best_order = results_df.loc[0, 'order']

# 4. 최적 ARIMA 모델 학습
model = ARIMA(log_return, order=best_order)
model_fit = model.fit()
print(model_fit.summary())

# 5. 향후 10일 수익률 예측
forecast_result = model_fit.get_forecast(steps=10)
mean_forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

print("\nARIMA 예측 수익률 (다음 10거래일):")
for i in range(10):
    print(f"Day {i+1}: 예상 수익률 = {mean_forecast.iloc[i]:.5f}, "
          f"95% CI = [{conf_int.iloc[i,0]:.5f}, {conf_int.iloc[i,1]:.5f}]")

# 6. GARCH(1,1)로 변동성 예측
resid = model_fit.resid
garch_model = arch_model(resid, vol='GARCH', p=1, q=1, mean='Zero')
garch_fit = garch_model.fit(disp='off')

forecast_vol = garch_fit.forecast(horizon=10)
predicted_variance = forecast_vol.variance.values[-1]
predicted_volatility = np.sqrt(predicted_variance)

print("\n예측된 향후 10일 변동성 (표준편차):")
for i, v in enumerate(predicted_volatility, 1):
    print(f"Day {i}: {v:.6f}")

# 7. 수익률 → 예측 가격
cum_returns = np.cumsum(mean_forecast)
last_price = aapl_close.iloc[-1]
predicted_prices = last_price * np.exp(cum_returns)

# 신뢰구간 계산 (±1.96 * σ)
upper_band = last_price * np.exp(cum_returns + 1.96 * predicted_volatility)
lower_band = last_price * np.exp(cum_returns - 1.96 * predicted_volatility)

# 8. 시각화
plt.figure(figsize=(12, 6))
days = np.arange(1, 11)

plt.plot(days, predicted_prices, label='Predicted Price Path', color='blue', linewidth=2)
plt.fill_between(days, lower_band, upper_band, color='gray', alpha=0.3, label='95% Prediction Interval')
plt.axhline(y=last_price, color='black', linestyle='--', label='Current Price')

plt.xlabel('Future Days')
plt.ylabel('Price')
plt.title('10-Day Price Forecast with ARIMA-GARCH and Volatility Bands')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
