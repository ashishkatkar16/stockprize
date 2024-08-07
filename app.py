import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Function to fetch and process stock data
def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Avg Price'] = data[['Open', 'Low', 'High', 'Adj Close']].mean(axis=1)
    data['Differenced'] = data['Avg Price'].diff()
    data.dropna(subset=['Differenced'], inplace=True)
    return data


# Function to fit SARIMAX model and make predictions
def sarimax_model(train, test, steps):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Seasonal order can be tuned
    fitted_model = model.fit(disp=False)
    predictions = fitted_model.forecast(steps=len(test))
    future_predictions = fitted_model.forecast(steps=steps)

    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)

    return predictions, future_predictions, mae, mse, rmse


# Streamlit app
st.title('Stock Price Prediction with SARIMAX')

# Inputs
symbol = st.text_input('Stock Symbol', 'RELIANCE.NS')
start_date = st.date_input('Start Date', pd.to_datetime('2014-07-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-06-30'))
future_steps = st.number_input('Future Steps', min_value=1, max_value=100, value=30)

# Fetch and display stock data
data = get_stock_data(symbol, start_date, end_date)
if data.empty:
    st.error('No data fetched. Please check the stock symbol and date range.')
else:
    train_size = int(len(data) * 0.8)
    train = data['Avg Price'][:train_size]
    test = data['Avg Price'][train_size:]

    # Fit SARIMAX model and get predictions
    sarimax_predictions, sarimax_future, sarimax_mae, sarimax_mse, sarimax_rmse = sarimax_model(train, test,
                                                                                                future_steps)

    # Display evaluation metrics
    st.write(f"SARIMAX Model - MAE: {sarimax_mae}, MSE: {sarimax_mse}, RMSE: {sarimax_rmse}")

    # Generate future dates
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='B')


    # Plot the results
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['Avg Price'], label='Historical Data')
    ax.plot(test.index, test, label='Test Data', color='blue')
    ax.plot(test.index, sarimax_predictions, label='SARIMAX Predictions', color='red')
    ax.plot(future_dates, sarimax_future, label='SARIMAX Future Predictions', color='orange')
    ax.legend()
    st.pyplot(fig)

    st.write('## Future Predictions')
    sm = pd.DataFrame({'future_dates':future_dates,'Avg Predicted Price':sarimax_future})
    st.write(sm.reset_index().drop(columns= ['index']))

    # Future Predictions
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(future_dates, sarimax_future, label='SARIMAX Future Predictions', color='cyan', linestyle='--')
    ax.legend()
    st.pyplot(fig)
