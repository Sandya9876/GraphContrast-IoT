import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

LABEL_CANDIDATES = ['label','Label','attack','Attack','anomaly','Anomaly','class','Class','Normal/Attack','normal_attack']


def infer_label_column(df, preferred=None):
    if preferred and preferred in df.columns: return preferred
    for c in LABEL_CANDIDATES:
        if c in df.columns: return c
    return None


def label_to_binary(s):
    if s is None: return None
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0).astype(float).values > 0).astype(int)
    vals = s.astype(str).str.lower().str.strip()
    normal_tokens = {'normal','0','false','benign','no','none'}
    return (~vals.isin(normal_tokens)).astype(int).values


def clean_dataframe(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().interpolate(limit_direction='both')
    return df


def select_numeric_features(df, label_col=None):
    drop_cols = [] if label_col is None else [label_col]
    tmp = df.drop(columns=drop_cols, errors='ignore')
    numeric = tmp.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        for c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
        numeric = tmp.select_dtypes(include=[np.number]).copy()
    return numeric


def scale_features(train, val, test, method='standard'):
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    return train_s.astype('float32'), val_s.astype('float32'), test_s.astype('float32'), scaler
