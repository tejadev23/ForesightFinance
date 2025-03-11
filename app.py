from flask import Flask, render_template, request
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_ticker = request.form['stock']
    days = int(request.form['days'])

    # Fetch the stock data
    stock_data = yf.download(stock_ticker, period="5y")
    stock_data['Date'] = stock_data.index

    # Prepare data for RandomForest
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Date_ordinal'] = pd.to_datetime(stock_data['Date']).map(pd.Timestamp.toordinal)

    X = stock_data[['Date_ordinal']].values
    y = stock_data['Close'].values

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict the stock prices for future days
    future_dates = np.array([stock_data['Date'].max() + pd.Timedelta(days=i) for i in range(1, days+1)])
    future_dates_ordinal = pd.to_datetime(future_dates).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    
    predicted_prices = model.predict(future_dates_ordinal)

    # Create a human-readable prediction output
    prediction_output = {}
    for i in range(days):
        prediction_output[str(future_dates[i].date())] = round(predicted_prices[i], 2)  # Rounding for better readability

    return render_template('index.html', prediction=prediction_output)

if __name__ == '__main__':
    app.run(debug=True)
#.\venv\Scripts\activate