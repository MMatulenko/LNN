# LNN

This repository demonstrates Liquid Neural Networks (LNNs) using the [ncps](https://github.com/mlech26l/ncps) library. Liquid Neural Networks are continuous-time neural models that maintain a small number of trainable parameters while exhibiting rich temporal dynamics. They can learn complex time-series patterns without needing huge model sizes.

## Using the example notebook
`LNN_robust_model.ipynb` shows a minimal example that trains an LNN to fit a sinusoidal sequence with irregular sampling. It installs `ncps`, builds a Liquid Time-Constant (LTC) model, trains it, and visualises the results.

## Example script
`lnn_timeseries_product.py` provides a simple script that illustrates a potential real-world use of LNNs: anomaly detection in vibration data. The script generates synthetic sensor data, trains an LTC network, then flags anomalies based on prediction error.

Run the script with:
```bash
pip install numpy tensorflow ncps
python lnn_timeseries_product.py
```

## Stock prediction pipeline
`lnn_stock_pipeline.py` trains an LNN to forecast stock closing prices.
It expects a CSV file with `Date` and `Close` columns. A small synthetic
dataset `sample_stocks.csv` is included for testing.

```bash
pip install tensorflow ncps pandas
# synthetic data
python lnn_stock_pipeline.py --csv sample_stocks.csv --epochs 5
# or fetch real Bitcoin prices
python lnn_stock_pipeline.py --btc --epochs 5
```

The GitHub Actions workflow in `.github/workflows/stock.yml` installs the
dependencies and runs the pipeline automatically on each push.

## Potential applications
Liquid Neural Networks have been explored in robotics, autonomous vehicles, and other domains requiring robust predictions from sequential data. Their small parameter count makes them attractive for embedded systems and edge devices where compute resources are limited.

For more information see the official documentation: <https://ncps.readthedocs.io/en/latest/index.html>
