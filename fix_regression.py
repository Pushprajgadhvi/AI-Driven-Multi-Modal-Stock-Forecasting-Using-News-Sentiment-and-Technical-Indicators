"""
Fix regression models with consistent data sizing
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load merged data
merged = pd.read_csv('reliance_final.csv')

# Define split
split_index = int(0.8 * len(merged))

# Get Close prices
y_train_reg = merged['Close'].iloc[:split_index].values
y_test_reg = merged['Close'].iloc[split_index:].values

# Load feature data - use the same size as y_test
# For simplicity, we'll create features from Close, Volume, and sentiment
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA20', 'EMA50', 
                'RSI', 'MACD', 'MACD_signal', 'ATR', 'Volatility', 'Volume_MA', 
                'sentiment_score', 'sentiment_num']

X_data = merged[feature_cols].fillna(merged[feature_cols].mean()).values
X_train_reg = X_data[:split_index]
X_test_reg = X_data[split_index:split_index+len(y_test_reg)]

# Remove NaN from y values
mask_train = ~np.isnan(y_train_reg)
mask_test = ~np.isnan(y_test_reg)

X_train_reg = X_train_reg[mask_train]
y_train_reg = y_train_reg[mask_train]

X_test_reg = X_test_reg[mask_test]
y_test_reg = y_test_reg[mask_test]

print(f"X_train shape: {X_train_reg.shape}")
print(f"X_test shape: {X_test_reg.shape}")
print(f"y_train shape: {y_train_reg.shape}")
print(f"y_test shape: {y_test_reg.shape}")

# ------- SVR -------
print("\nTraining SVR...")
model_svr = SVR(kernel='rbf', C=100, gamma=0.01)
model_svr.fit(X_train_reg, y_train_reg)
y_pred_svr = model_svr.predict(X_test_reg)
svr_r2 = r2_score(y_test_reg, y_pred_svr)
svr_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_svr))

# ------- Ridge -------
print("Training Ridge...")
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train_reg, y_train_reg)
y_pred_ridge = model_ridge.predict(X_test_reg)
ridge_r2 = r2_score(y_test_reg, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge))

# ------- Gradient Boosting -------
print("Training GBR...")
model_gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
model_gbr.fit(X_train_reg, y_train_reg)
y_pred_gbr = model_gbr.predict(X_test_reg)
gbr_r2 = r2_score(y_test_reg, y_pred_gbr) 
gbr_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_gbr))

# Print results
print("\n" + "="*60)
print("REGRESSION MODELS - Price Prediction")
print("="*60)
print(f"\nSVR (Support Vector Regression):")
print(f"  R² Score: {svr_r2:.4f}")
print(f"  RMSE: ₹{svr_rmse:.2f}")
print(f"\nRidge Regression:")
print(f"  R² Score: {ridge_r2:.4f}")
print(f"  RMSE: ₹{ridge_rmse:.2f}")
print(f"\nGradient Boosting Regressor:")
print(f"  R² Score: {gbr_r2:.4f}")
print(f"  RMSE: ₹{gbr_rmse:.2f}")
print("="*60)

# Save results
results = {
    'svr_r2': svr_r2,
    'svr_rmse': svr_rmse,
    'ridge_r2': ridge_r2,
    'ridge_rmse': ridge_rmse,
    'gbr_r2': gbr_r2,
    'gbr_rmse': gbr_rmse,
    'y_pred_svr': y_pred_svr,
    'y_pred_ridge': y_pred_ridge,
    'y_pred_gbr': y_pred_gbr,
    'y_test_reg': y_test_reg
}

np.save('regression_results.npy', results, allow_pickle=True)
print("\n✓ Results saved to regression_results.npy")
