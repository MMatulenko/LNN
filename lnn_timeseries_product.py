# Simple example: Anomaly detection for vibration data using Liquid Neural Networks
import numpy as np
from tensorflow import keras
from ncps import wirings
from ncps.keras import LTC

# Generate synthetic vibration data (sine wave with noise and occasional spikes)
N = 500
time = np.linspace(0, 10*np.pi, N)
vibration = np.sin(time) + 0.05 * np.random.randn(N)
# Introduce anomalies
vibration[150] += 3
vibration[350] -= 3

data_x = vibration[:-1].reshape(1, -1, 1).astype(np.float32)
# next-step prediction target
data_y = vibration[1:].reshape(1, -1, 1).astype(np.float32)

# Split into train and test
train_ratio = 0.8
split = int((N-1)*train_ratio)
train_x, test_x = data_x[:, :split, :], data_x[:, split:, :]
train_y, test_y = data_y[:, :split, :], data_y[:, split:, :]

# Build a simple LTC network
wiring = wirings.AutoNCP(12,1)
model = keras.models.Sequential([
    LTC(wiring, return_sequences=True, name="ltc", input_shape=(None, 1)),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# Train the model
model.fit(x=train_x, y=train_y, epochs=10, verbose=0)

# Evaluate and flag anomalies in the test set
pred = model.predict(test_x, verbose=0)[0,:,0]
true = test_y[0,:,0]
error = np.abs(pred - true)

threshold = np.mean(error) + 3*np.std(error)
anomalies = np.where(error > threshold)[0]
print("Anomaly indices in test set:", anomalies)
