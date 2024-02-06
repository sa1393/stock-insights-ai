import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

# 데이터 다운로드
start_date = '2006-01-01'
end_date = '2018-01-01'
df = yf.download('329180.KS')

# 데이터 전처리
sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(df.iloc[:,1:2].values)

# 훈련 데이터 준비
X_train = []
y_train = []
for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# LSTM 모델 구축

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=30))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

# 모델 학습
regressor.fit(X_train, y_train, epochs=50, batch_size=32)

# 마지막 60일 데이터로 다음날 주가 예측
last_60_days = scaled_data[-60:]
last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
predicted_stock_price = regressor.predict(last_60_days)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# 예측된 주가 출력
print("다음날 예측 주가: ", predicted_stock_price[0,0])