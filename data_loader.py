import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

SEED = 42

def load_and_preprocess(data_path='D;/Rensi_DP_FL/data set.csv'):
    df = pd.read_csv(data_path)
    label_col = df.columns[-1]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != label_col]
    X_raw = df[numeric_cols].values.astype('float32')
    y_raw = df[label_col].values
    col_means = np.nanmean(X_raw, axis=0)
    inds = np.where(np.isnan(X_raw))
    if inds[0].size > 0:
        X_raw[inds] = np.take(col_means, inds[1])
    if not np.issubdtype(y_raw.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.astype('int64')
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=SEED, stratify=y_trainval)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_clients(X_train, y_train, num_clients=10, cap=None):
    n = X_train.shape[0]
    idx = np.arange(n)
    np.random.seed(SEED)
    np.random.shuffle(idx)
    shards = np.array_split(idx, num_clients)
    clients = []
    for s in shards:
        sel = s
        if cap is not None and len(sel) > cap:
            sel = np.random.choice(sel, size=cap, replace=False)
        clients.append((X_train[sel], y_train[sel]))
    return clients

def create_clients_dirichlet(X_train, y_train, num_clients=10, alpha=0.5, cap=None):
    """
    Non-IID partitioning of data across clients using a Dirichlet distribution
    over label proportions.

    Each class's samples are split across clients according to a Dirichlet(α)
    draw, then concatenated. Smaller alpha => more skew / non-IID.
    """
    np.random.seed(SEED)
    num_classes = int(np.max(y_train)) + 1

    # List of indices per client
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)

        # Dirichlet proportions for this class across clients
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Turn proportions into split indices
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)

        for cid, idx_split in enumerate(splits):
            client_indices[cid].extend(idx_split.tolist())

    clients = []
    for cid_idx in client_indices:
        sel = np.array(cid_idx, dtype=int)
        if cap is not None and len(sel) > cap:
            sel = np.random.choice(sel, size=cap, replace=False)
        clients.append((X_train[sel], y_train[sel]))
    return clients

