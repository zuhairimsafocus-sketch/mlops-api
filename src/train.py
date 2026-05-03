import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lightgbm import LGBMRegressor


# ================================
# LOAD CONFIG
# ================================
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data"]["path"]
TEST_SIZE = config["train"]["test_size"]
RANDOM_STATE = config["train"]["random_state"]
MODEL_PARAMS = config["model"]["params"]


# ================================
# LOAD DATA
# ================================
df = pd.read_csv(DATA_PATH)

# pilih features (ikut notebook awak)
FEATURES = ['Model', 'Fuel Type', 'Turbo', 'Horsepower']
TARGET = 'Base Price (USD)'

X = df[FEATURES]
y = df[TARGET]


# ================================
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ================================
# PREPROCESSING
# ================================
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', num_cols)
])


# ================================
# MODEL
# ================================
model = LGBMRegressor(**MODEL_PARAMS)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])


# ================================
# MLFLOW SETUP
# ================================
mlflow.set_experiment("car_price_prediction")

with mlflow.start_run():

    # ===== TRAIN =====
    pipeline.fit(X_train, y_train)

    # ===== PREDICT =====
    y_pred = pipeline.predict(X_test)

    # ===== METRICS =====
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ===== LOG PARAMS =====
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    # log semua hyperparameters
    for key, value in MODEL_PARAMS.items():
        mlflow.log_param(key, value)

    # ===== LOG METRICS =====
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # ===== LOG MODEL =====
    mlflow.sklearn.log_model(pipeline, "model")

    print("✅ Training complete")
    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"R2  : {r2}")


# ================================
# SAVE MODEL (LOCAL)
# ================================
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model.pkl")

print("✅ Model saved at models/model.pkl")