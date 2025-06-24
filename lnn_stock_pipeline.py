import argparse
import pandas as pd
import numpy as np
from tensorflow import keras
from ncps import wirings
from ncps.keras import LTC

parser = argparse.ArgumentParser(description="Train an LNN on stock data")
parser.add_argument('--csv', default='sample_stocks.csv', help='Path to CSV with Date and Close columns')
parser.add_argument('--btc', action='store_true', help='Use real BTC price data instead of --csv')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lookback', type=int, default=5)
args = parser.parse_args()

# Load data
if args.btc:
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    df = pd.read_csv(url, usecols=["time", "PriceUSD"])
    df = df.dropna()
    prices = df["PriceUSD"].astype(np.float32).values
    # limit to most recent 1000 days to keep training fast
    prices = prices[-1000:]
else:
    csv = pd.read_csv(args.csv)
    prices = csv['Close'].values.astype(np.float32)

# Build sequences
lookback = args.lookback
X, y = [], []
for i in range(len(prices) - lookback):
    X.append(prices[i:i+lookback])
    y.append(prices[i+lookback])
X = np.array(X).reshape(-1, lookback, 1)
y = np.array(y).reshape(-1, 1)

# Train/test split
split = int(0.8 * len(X))
train_x, test_x = X[:split], X[split:]
train_y, test_y = y[:split], y[split:]

# Build LNN model
wiring = wirings.AutoNCP(16, 1)
model = keras.models.Sequential([
    LTC(wiring, return_sequences=False, input_shape=(lookback, 1)),
    keras.layers.Dense(1)
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])

model.fit(train_x, train_y, epochs=args.epochs, verbose=0)
loss, mae = model.evaluate(test_x, test_y, verbose=0)
print(f"Test MAE: {mae:.4f}")
