import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import hopsworks
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

logger.info("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

logger.info("Loading feature group 'aqi_features' (version 1)...")
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()
logger.info(f"Data loaded — shape: {df.shape}")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns
df = df[(df[numeric_cols] <= 1e6).all(axis=1)]

target_col = "aqi_aqicn"
feature_cols = [c for c in numeric_cols if c != target_col]
df.dropna(subset=[target_col] + feature_cols, inplace=True)

if df.empty:
    raise ValueError("Dataset is empty after cleaning!")

SEQUENCE_LENGTH = 7
X_seq, y_seq = [], []

for i in range(SEQUENCE_LENGTH, len(df)):
    X_seq.append(df[feature_cols].iloc[i - SEQUENCE_LENGTH:i].values)
    y_seq.append(df[target_col].iloc[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)
logger.info(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

scaler_X = StandardScaler()
scaler_y = MinMaxScaler()

nsamples, ntimesteps, nfeatures = X_seq.shape
X_scaled = scaler_X.fit_transform(X_seq.reshape(nsamples * ntimesteps, nfeatures)).reshape(nsamples, ntimesteps, nfeatures)
y_scaled = scaler_y.fit_transform(y_seq.reshape(-1, 1))

split_idx = int(0.8 * len(X_scaled))
X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

logger.info("Building LSTM model...")
model = Sequential([
    LSTM(64, activation="tanh", return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))),
    Dropout(0.2),
    LSTM(32, activation="tanh"),
    Dropout(0.1),
    Dense(16, activation="relu"),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])

logger.info("Training LSTM model...")
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,          
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

metrics = {
    "Train_Loss": float(history.history["loss"][-1]),
    "Val_Loss": float(history.history["val_loss"][-1]),
    "Val_MAE": float(history.history["mae"][-1])
}
logger.info(f"Model Performance: {metrics}")

logger.info("Retraining on full dataset for deployment...")
model.fit(X_scaled, y_scaled, validation_split=0.1, epochs=20, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=0)

MODEL_NAME = "tf_lstm_aqi_model"
MODEL_DIR = os.path.join("models", MODEL_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(os.path.join(MODEL_DIR, "model.keras"))
joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.joblib"), compress=3)
joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.joblib"), compress=3)

metadata = {
    "model_name": MODEL_NAME,
    "trained_at": datetime.now().isoformat(),
    "sequence_length": SEQUENCE_LENGTH,
    "features_used": feature_cols,
    "target": target_col,
    "metrics": metrics
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

logger.info(f"Model, scalers, and metadata saved locally at: {MODEL_DIR}")

logger.info("Uploading model to Hopsworks Model Registry...")
try:
    try:
        model_meta = mr.get_model(MODEL_NAME)
        logger.info(f"Model '{MODEL_NAME}' exists. Updating files...")
        model_meta.update(MODEL_DIR)
    except Exception:
        logger.info(f"Model '{MODEL_NAME}' does not exist. Creating new entry...")
        model_meta = mr.python.create_model(
            name=MODEL_NAME,
            metrics=metrics,
            description="TensorFlow LSTM model for Karachi AQI forecasting"
        )
        model_meta.save(MODEL_DIR)
    logger.info("Model successfully saved/updated in Hopsworks.")
except Exception as e:
    logger.warning(f"Upload failed — model saved locally only. Error: {e}")

logger.info("Training pipeline completed successfully!")
