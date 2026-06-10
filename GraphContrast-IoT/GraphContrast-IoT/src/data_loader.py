from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocessing import clean_dataframe, infer_label_column, label_to_binary, select_numeric_features, scale_features
from .windowing import make_windows


def _synthetic_iot(n=6000, sensors=16, anomaly_ratio=0.08, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    X = []
    for i in range(sensors):
        X.append(np.sin(0.01*t*(1+i/20)) + 0.3*np.sin(0.03*t+i) + rng.normal(0,0.05,n))
    X = np.vstack(X).T
    y = np.zeros(n, dtype=int)
    starts = rng.choice(np.arange(200,n-200), size=max(1,int(n*anomaly_ratio/50)), replace=False)
    for s in starts:
        length = rng.integers(20,80); cols = rng.choice(sensors, size=rng.integers(2,6), replace=False)
        X[s:s+length, cols] += rng.normal(2.5,0.5,(length,len(cols)))
        y[s:s+length] = 1
    return pd.DataFrame(X, columns=[f'S{i+1}' for i in range(sensors)]), y


def load_dataset(cfg, seed=42):
    dcfg = cfg['dataset']; raw_dir = Path(dcfg['raw_dir'])
    files = sorted(glob.glob(str(raw_dir / dcfg.get('file_pattern','*.csv'))))
    frames = []
    if files:
        for f in files:
            try: frames.append(pd.read_csv(f))
            except Exception: frames.append(pd.read_csv(f, encoding='latin1'))
        df = clean_dataframe(pd.concat(frames, ignore_index=True))
        label_col = infer_label_column(df, dcfg.get('label_column'))
        y = label_to_binary(df[label_col]) if label_col else np.zeros(len(df), dtype=int)
        Xdf = select_numeric_features(df, label_col)
    elif dcfg.get('allow_synthetic', False):
        Xdf, y = _synthetic_iot(seed=seed)
    else:
        raise FileNotFoundError(f'No CSV files found in {raw_dir}.')

    Xdf = clean_dataframe(Xdf).fillna(0)
    feature_names = list(Xdf.columns)
    X = Xdf.values.astype('float32')

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=dcfg.get('test_size',0.3), shuffle=False)
    train_idx, val_idx = train_test_split(train_idx, test_size=dcfg.get('val_size',0.15), shuffle=False)
    Xtr, Xv, Xte = X[train_idx], X[val_idx], X[test_idx]
    ytr, yv, yte = y[train_idx], y[val_idx], y[test_idx]
    Xtr, Xv, Xte, scaler = scale_features(Xtr, Xv, Xte, dcfg.get('normalize','standard'))
    W, stride = dcfg.get('window_size',60), dcfg.get('stride',10)
    return {
        'train': make_windows(Xtr, ytr, W, stride),
        'val': make_windows(Xv, yv, W, stride),
        'test': make_windows(Xte, yte, W, stride),
        'feature_names': feature_names,
        'scaler': scaler,
    }
