"""Generate test fixtures for cross-runtime parity testing.

Trains small XGBoost models in Python and saves:
- Model files (UBJ format)
- Input data (JSON)
- Expected predictions (JSON)

Run: conda run -n prob python test/fixtures/generate.py
"""
import json
import numpy as np
import xgboost as xgb

print(f'xgboost {xgb.__version__}')

fixtures_dir = 'test/fixtures'


def save_fixture(name, X, y, params, num_round, preds):
    """Save a single fixture: model + data + expected predictions."""
    # Save input data
    with open(f'{fixtures_dir}/{name}.data.json', 'w') as f:
        json.dump({
            'X': X.tolist(),
            'y': y.tolist(),
            'params': params,
            'num_round': num_round,
            'predictions': preds.tolist()
        }, f)
    print(f'  {name}: {len(X)} samples, {num_round} rounds, {len(preds)} predictions')


# ============================================================
# 1. Regression (reg:squarederror)
# ============================================================
print('\n=== reg:squarederror ===')
np.random.seed(42)
X = np.random.rand(20, 2).astype(np.float32) * 10
y = (X[:, 0] + X[:, 1]).astype(np.float32)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'eta': 0.3,
    'verbosity': 0,
    'seed': 42
}
dtrain = xgb.DMatrix(X, label=y)
model = xgb.train(params, dtrain, num_boost_round=20)
preds = model.predict(dtrain)

# Save model as UBJ
model.save_model(f'{fixtures_dir}/regression.ubj')
save_fixture('regression', X, y, params, 20, preds)

# ============================================================
# 2. Binary classification (binary:logistic)
# ============================================================
print('\n=== binary:logistic ===')
np.random.seed(42)
X = np.random.rand(30, 2).astype(np.float32) * 10
y = (X[:, 0] + X[:, 1] > 10).astype(np.float32)

params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'eta': 0.3,
    'verbosity': 0,
    'seed': 42
}
dtrain = xgb.DMatrix(X, label=y)
model = xgb.train(params, dtrain, num_boost_round=20)
preds = model.predict(dtrain)

model.save_model(f'{fixtures_dir}/binary.ubj')
save_fixture('binary', X, y, params, 20, preds)

# ============================================================
# 3. Multiclass (multi:softprob)
# ============================================================
print('\n=== multi:softprob ===')
np.random.seed(42)
X = np.random.rand(30, 2).astype(np.float32) * 10
sums = X[:, 0] + X[:, 1]
y = np.where(sums < 7, 0, np.where(sums < 13, 1, 2)).astype(np.float32)

params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 3,
    'eta': 0.3,
    'verbosity': 0,
    'seed': 42
}
dtrain = xgb.DMatrix(X, label=y)
model = xgb.train(params, dtrain, num_boost_round=20)
preds = model.predict(dtrain)  # shape: (30, 3)

model.save_model(f'{fixtures_dir}/multiclass.ubj')
save_fixture('multiclass', X, y, params, 20, preds.flatten())

# ============================================================
# 4. Inference-only: load model and predict on new data
# ============================================================
print('\n=== Inference-only test ===')
np.random.seed(99)
X_test = np.random.rand(5, 2).astype(np.float32) * 10

# Load regression model and predict
model = xgb.Booster()
model.load_model(f'{fixtures_dir}/regression.ubj')
dtest = xgb.DMatrix(X_test)
preds_test = model.predict(dtest)

with open(f'{fixtures_dir}/inference.data.json', 'w') as f:
    json.dump({
        'X': X_test.tolist(),
        'predictions': preds_test.tolist(),
        'model': 'regression.ubj'
    }, f)
print(f'  inference: {len(X_test)} samples, {len(preds_test)} predictions')

print('\nDone!')
