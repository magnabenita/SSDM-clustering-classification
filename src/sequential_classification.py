# src/sequential_classification.py
"""
Sequential Classification Implementations with Visualizations:

1. Sequence-based Random Forest
2. Hidden Markov Model (HMM) Classifier
3. Conditional Random Field (CRF) Sequence Labeler
4. LSTM (RNN) Classifier
5. k-NN with DTW distance

Visualizations:
- Line plots of sequences colored by predicted class
- Heatmaps of sequence features
- DTW distance matrix heatmap (for k-NN)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Optional imports
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    import sklearn_crfsuite
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False

KERAS_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Masking
    from tensorflow.keras.utils import to_categorical
    KERAS_AVAILABLE = True
except ImportError:
    pass

try:
    from tslearn.neighbors import KNeighborsTimeSeriesClassifier
    from tslearn.metrics import cdist_dtw
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False

RESULTS_DIR = "results/sequential_classification"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------- Helpers --------------------
def magnitude_to_label(mag):
    if np.isnan(mag): return 0
    if mag < 2.5: return 0
    elif mag < 5.0: return 1
    else: return 2

def build_sequences(df, window=5):
    df = df.sort_values("time").reset_index(drop=True)
    mags = df["magnitude"].values
    depths = df["depth"].fillna(0).values
    X, y = [], []
    for i in range(len(df)-window):
        feat = [
            np.nanmean(mags[i:i+window]),
            np.nanstd(mags[i:i+window]),
            np.nanmax(mags[i:i+window]),
            np.nanmean(depths[i:i+window]),
            np.nanstd(depths[i:i+window])
        ]
        X.append(feat)
        y.append(magnitude_to_label(mags[i+window]))
    return np.array(X), np.array(y)

def save_classification_report(report, model_name):
    with open(os.path.join(RESULTS_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(report)

def save_predictions(y_true, y_pred, model_name):
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        os.path.join(RESULTS_DIR, f"{model_name}_predictions.csv"), index=False
    )

# -------------------- Visualization --------------------
def plot_sequence_heatmap(X, y_pred, model_name, feature_idx=0):
    # Convert X to 2D numeric array
    if isinstance(X, list):
        if isinstance(X[0], dict):  # CRF
            X_plot = np.array([[float(f["mag"]) for f in seq] for seq in X])
        else:  # HMM
            X_plot = np.array([seq[:, feature_idx].flatten() for seq in X])
    elif X.ndim == 3:  # LSTM / DTW k-NN
        X_plot = X[:, :, feature_idx]
    else:
        X_plot = X

    plt.figure(figsize=(12,6))
    sns.heatmap(X_plot, cmap="viridis")
    plt.title(f"{model_name} Sequence Heatmap (feature {feature_idx})")
    plt.xlabel("Time / Feature Index")
    plt.ylabel("Sequence Index")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_heatmap.png"))
    plt.close()

def plot_sequences_line(X, y_pred, model_name):
    X_plot = np.array(X)
    if X_plot.ndim == 3:
        X_plot = X_plot[:, :, 0]  # magnitude
    plt.figure(figsize=(12,6))
    for idx, seq in enumerate(X_plot):
        plt.plot(seq, label=f"Seq {idx} (Class {y_pred[idx]})", alpha=0.7)
    plt.title(f"{model_name} - Sequence Line Plot")
    plt.xlabel("Time / Feature Index")
    plt.ylabel("Magnitude / Value")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_lineplot.png"))
    plt.close()

def plot_dtw_distance_matrix(X, model_name):
    if not TSLEARN_AVAILABLE: return
    X_seq = np.array(X).reshape((len(X), -1))
    dist_matrix = cdist_dtw(X_seq)
    plt.figure(figsize=(10,8))
    sns.heatmap(dist_matrix, cmap="magma")
    plt.title(f"{model_name} - DTW Distance Matrix")
    plt.xlabel("Sequence Index")
    plt.ylabel("Sequence Index")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_dtw_matrix.png"))
    plt.close()

# -------------------- 1. Random Forest --------------------
def sequence_random_forest(df, window=5):
    X, y = build_sequences(df, window)
    if len(X)==0: return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    report = classification_report(y_test, preds)
    print("Random Forest Classification Report:\n", report)
    save_classification_report(report, "Random_Forest")
    save_predictions(y_test, preds, "Random_Forest")
    
    plot_sequence_heatmap(X_test, preds, "Random_Forest")
    plot_sequences_line(X_test, preds, "Random_Forest")
    
    return clf

# -------------------- 2. Hidden Markov Model --------------------
def hmm_classifier(df, window=10, n_states=3):
    if not HMM_AVAILABLE: return None
    df = df.sort_values("time").reset_index(drop=True)
    mags = df["magnitude"].values
    seqs, labels = [], []
    for i in range(len(df)-window):
        seqs.append(mags[i:i+window].reshape(-1,1))
        labels.append(magnitude_to_label(mags[i+window]))
    labels = np.array(labels)

    models = {}
    for c in np.unique(labels):
        cls_seqs = [s for s,l in zip(seqs, labels) if l==c]
        if len(cls_seqs)<2: continue
        X_concat = np.vstack(cls_seqs)
        lengths = [len(s) for s in cls_seqs]
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
        model.fit(X_concat, lengths)
        models[c] = model

    y_pred = [max({c:m.score(s) for c,m in models.items()}, key=lambda k: {c:m.score(s) for c,m in models.items()}[k]) for s in seqs]
    report = classification_report(labels, y_pred)
    print("HMM Classification Report:\n", report)
    save_classification_report(report, "HMM")
    save_predictions(labels, y_pred, "HMM")

    plot_sequence_heatmap(seqs, y_pred, "HMM")
    plot_sequences_line(np.array(seqs).reshape(len(seqs), window, 1), y_pred, "HMM")
    
    return models

# -------------------- 3. Conditional Random Field --------------------
def crf_classifier(df, window=5):
    if not CRF_AVAILABLE: return None
    sequences, labels = [], []
    df = df.sort_values("time").reset_index(drop=True)
    for i in range(0, len(df)-window, window):
        chunk = df.iloc[i:i+window]
        feats, labs = [], []
        for _, row in chunk.iterrows():
            feats.append({"mag": str(row["magnitude"]),
                          "depth": str(row.get("depth",0)),
                          "lat": str(row.get("latitude",0))})
            labs.append(str(magnitude_to_label(row["magnitude"])))
        sequences.append(feats)
        labels.append(labs)

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", max_iterations=100)
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)
    
    y_true_flat = [int(l) for seq in y_test for l in seq]
    y_pred_flat = [int(l) for seq in y_pred for l in seq]
    
    report = classification_report(y_true_flat, y_pred_flat)
    print("CRF Classification Report:\n", report)
    save_classification_report(report, "CRF")
    save_predictions(y_true_flat, y_pred_flat, "CRF")
    
    plot_sequence_heatmap(X_test, y_pred_flat, "CRF")
    
    return crf

# -------------------- 4. LSTM --------------------
def lstm_classifier(df, window=10, epochs=10, batch_size=32):
    if not KERAS_AVAILABLE: return None
    df = df.sort_values("time").reset_index(drop=True)
    mags = df["magnitude"].fillna(0).values
    depths = df["depth"].fillna(0).values
    X, y = [], []
    for i in range(len(df)-window):
        X.append(np.vstack([mags[i:i+window], depths[i:i+window]]).T)
        y.append(magnitude_to_label(mags[i+window]))
    X = np.array(X)
    y = np.array(y)
    if len(X)==0: return None
    
    y_cat = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    model = Sequential([
        Masking(mask_value=0., input_shape=(X.shape[1], X.shape[2])),
        LSTM(64),
        Dense(y_cat.shape[1], activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    
    preds = model.predict(X_test)
    preds_lbl = np.argmax(preds, axis=1)
    y_test_lbl = np.argmax(y_test, axis=1)
    
    report = classification_report(y_test_lbl, preds_lbl)
    print("LSTM Classification Report:\n", report)
    save_classification_report(report, "LSTM")
    save_predictions(y_test_lbl, preds_lbl, "LSTM")
    
    plot_sequence_heatmap(X, np.argmax(y_cat, axis=1), "LSTM")
    plot_sequences_line(X, np.argmax(y_cat, axis=1), "LSTM")
    
    return model

# -------------------- 5. DTW k-NN --------------------
def dtw_knn_classifier(df, window=20, n_neighbors=3):
    if not TSLEARN_AVAILABLE: return None
    df = df.sort_values("time").reset_index(drop=True)
    mags = df["magnitude"].fillna(0).values
    X, y = [], []
    for i in range(len(df)-window):
        X.append(mags[i:i+window])
        y.append(magnitude_to_label(mags[i+window]))
    X = np.array(X).reshape((-1,window,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    
    report = classification_report(y_test, preds)
    print("DTW k-NN Classification Report:\n", report)
    save_classification_report(report, "DTW_kNN")
    save_predictions(y_test, preds, "DTW_kNN")
    
    plot_sequences_line(X.reshape(len(X), window), y, "DTW_kNN")
    plot_dtw_distance_matrix(X, "DTW_kNN")
    
    return knn
