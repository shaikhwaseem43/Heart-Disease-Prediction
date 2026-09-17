import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_csv(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    if not pd.api.types.is_numeric_dtype(y):
        labels = {v:i for i,v in enumerate(sorted(y.dropna().unique()))}
        y = y.map(labels)
    return X.astype(float), pd.Series(y).astype(int).to_numpy()

def split_scale(X, y, test_size=0.20, random_state=42):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler()
    return scaler.fit_transform(Xtr), scaler.transform(Xte), ytr, yte, scaler

def tabular_to_sequence(X):
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(X.shape[0], X.shape[1], 1)

def make_synthetic_tabular(n_samples=800, n_features=13, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    score = 1.2*X[:,0] - .9*X[:,1] + .7*X[:,2] + .4*X[:,3]
    p = 1/(1+np.exp(-score))
    y = (rng.random(n_samples) < p).astype(int)
    return pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)]), y
