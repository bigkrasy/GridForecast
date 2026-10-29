# GridForecast

Прогнозиране на потребление на електроенергия с ML.

## Резултати

| Model | MAPE |
|-------|:----:|
| LightGBM (tuned) | **2.91%** 🏆 |
| XGBoost | 2.97% |
| Prophet v3 | 6.84% |
| LSTM | 7.81% |
| SARIMA v2 | 8.07% |
| Seasonal Naive | 9.20% |

## Pipeline

1. EDA
2. Baselines
3. SARIMA
4. Prophet
5. Boosting (champion)
6. LSTM
7. Drift detection
8. Retraining

## Production

- **Model:** LightGBM (tuned)
- **Features:** 29 (lag, rolling, cyclical, exogenous)
- **Monitoring:** Daily residual check + PSI
- **Retraining:** Quarterly full retrain