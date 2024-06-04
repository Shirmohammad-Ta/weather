

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv('weather_data.csv', parse_dates=['Date'])


print(data.head())


def create_features(df, target_col='Temperature', window=7):
    for i in range(1, window + 1):
        df[f'Lag_{i}'] = df[target_col].shift(i)
    return df.dropna()

data = create_features(data)


X = data.drop(columns=['Date', 'Temperature'])
y = data['Temperature']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Temperature'], label='Actual Temperatures')
plt.plot(data['Date'][len(data) - len(y_test):], y_pred, label='Predicted Temperatures', color='red')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Weather Forecasting with Random Forest')
plt.legend()
plt.show()


feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("Feature Importances:")
print(feature_importances)


import joblib
joblib.dump(model, 'weather_forecast_model.pkl')


# model = joblib.load('weather_forecast_model.pkl')


def predict_future(model, recent_data, days=7):
    recent_data = recent_data[-days:]
    predictions = []
    for _ in range(days):
        if len(recent_data) < days:
            break
        features = recent_data[-days:].flatten()
        prediction = model.predict([features])[0]
        predictions.append(prediction)
        recent_data = np.append(recent_data[1:], prediction)
    return predictions


recent_temps = np.array(data['Temperature'][-7:])
future_predictions = predict_future(model, recent_temps, days=7)
print("Future Temperature Predictions:", future_predictions)
