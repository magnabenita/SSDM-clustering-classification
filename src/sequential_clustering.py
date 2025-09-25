"""
Sequential Clustering Implementations with Evaluation:
1. DTW + k-Means
2. PrefixSpan + k-Means
3. HMM Clustering
4. DBA-kMeans
5. Sequence Autoencoder + Latent Space Clustering
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    v_measure_score,
)
from scipy.spatial.distance import pdist
from hmmlearn.hmm import GaussianHMM

TSLEARN_AVAILABLE = False
PREFIXSPAN_AVAILABLE = False
HMM_AVAILABLE = False
KERAS_AVAILABLE = False

try:
    from tslearn.clustering import TimeSeriesKMeans
    TSLEARN_AVAILABLE = True
except Exception:
    pass

try:
    from prefixspan import PrefixSpan
    PREFIXSPAN_AVAILABLE = True
except Exception:
    pass

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except Exception:
    pass

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Input, RepeatVector
    KERAS_AVAILABLE = True
except Exception:
    pass


# Optional dependencies
try:
    from tslearn.clustering import TimeSeriesKMeans
    TSLEARN_AVAILABLE = True
except Exception:
    TSLEARN_AVAILABLE = False

try:
    from prefixspan import PrefixSpan
    PREFIXSPAN_AVAILABLE = True
except Exception:
    PREFIXSPAN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Input, RepeatVector
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# -------------------- Paths --------------------
RESULTS_DIR = "results/sequential_clustering"
os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------- Helpers --------------------
def build_sequences(df, window=10):
    """Convert earthquake data into sliding sequences (magnitude, depth)."""
    df = df.sort_values("time").reset_index(drop=True)
    mags = df["magnitude"].fillna(0).values
    depths = df["depth"].fillna(0).values

    sequences = []
    for i in range(len(df) - window):
        seq = np.vstack([mags[i : i + window], depths[i : i + window]]).T
        sequences.append(seq)
    return np.array(sequences)


def save_cluster_labels(labels, model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_clusters.csv")
    pd.DataFrame({"sequence_id": range(len(labels)), "cluster": labels}).to_csv(
        path, index=False
    )


def save_metrics(metrics, model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(path, index=False)

import matplotlib.pyplot as plt

# -------------------- Visualization --------------------
import matplotlib.pyplot as plt

def plot_cluster_representatives(sequences, labels, centroids, model_name):
    n_clusters = centroids.shape[0]
    timesteps = sequences.shape[1]

    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 4 * n_clusters), sharex=True)

    if n_clusters == 1:
        axes = [axes]

    for c in range(n_clusters):
        ax = axes[c]

        # Plot all sequences in this cluster (light gray)
        for seq in sequences[labels == c]:
            ax.plot(seq[:, 0], color="lightgray", alpha=0.5)  # magnitude
            # ax.plot(seq[:, 1], color="lightblue", alpha=0.3)  # depth (optional)

        # Plot centroid (red = magnitude, blue = depth)
        ax.plot(centroids[c, :, 0], color="red", linewidth=2, label="Centroid Magnitude")
        ax.plot(centroids[c, :, 1], color="blue", linewidth=2, label="Centroid Depth")

        # Label with trend description
        if centroids[c, -1, 0] > centroids[c, 0, 0]:
            trend = "Rising Magnitude"
        elif centroids[c, -1, 0] < centroids[c, 0, 0]:
            trend = "Decreasing Magnitude"
        else:
            trend = "Stable Magnitude"

        ax.set_title(f"Cluster {c+1}: {trend}", fontsize=14)
        ax.legend()

    plt.suptitle(f"Cluster Representatives - {model_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_representatives.png"))
    plt.show()

# -------------------- Evaluation Metrics --------------------
def dunn_index(X, labels):
    """Compute Dunn Index for clustering quality."""
    clusters = np.unique(labels)
    delta = np.inf
    big_delta = -np.inf

    # Inter-cluster distance
    for i in clusters:
        xi = X[labels == i]
        for j in clusters:
            if i >= j:
                continue
            xj = X[labels == j]
            dist = np.linalg.norm(xi[:, None] - xj, axis=2)
            delta = min(delta, dist.min())

    # Intra-cluster diameter
    for i in clusters:
        xi = X[labels == i]
        dist = pdist(xi)
        if len(dist) > 0:
            big_delta = max(big_delta, dist.max())

    return delta / big_delta


def evaluate_clustering(sequences, labels, true_labels=None):
    """Compute clustering evaluation metrics."""
    X = sequences.reshape(sequences.shape[0], -1)
    metrics = {
        "silhouette_score": silhouette_score(X, labels),
        "calinski_harabasz_index": calinski_harabasz_score(X, labels),
        "davies_bouldin_index": davies_bouldin_score(X, labels),
        "dunn_index": dunn_index(X, labels),
    }
    if true_labels is not None:
        metrics["fowlkes_mallows_index"] = fowlkes_mallows_score(true_labels, labels)
        metrics["v_measure"] = v_measure_score(true_labels, labels)
    return metrics


# -------------------- 1. DTW + k-Means --------------------
def dtw_kmeans_clustering(df, window=10, n_clusters=3, true_labels=None):
    if not TSLEARN_AVAILABLE:
        raise RuntimeError("tslearn not installed. Run: pip install tslearn")

    sequences = build_sequences(df, window)
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    labels = model.fit_predict(sequences)

    save_cluster_labels(labels, "dtw_kmeans")
    metrics = evaluate_clustering(sequences, labels, true_labels)
    save_metrics(metrics, "dtw_kmeans")
    centroids = model.cluster_centers_ 
    # Visualization
    plot_cluster_representatives(sequences, labels,centroids, "dtw_kmeans")

    return labels, metrics

# -------------------- 2. PrefixSpan + k-Means --------------------
def prefixspan_kmeans_clustering(df, window=10, n_clusters=3, true_labels=None):
    if not PREFIXSPAN_AVAILABLE:
        raise RuntimeError("prefixspan not installed. Run: pip install prefixspan")

    sequences = build_sequences(df, window)
    sequences_int = sequences[:, :, 0].astype(int)

    # Extract sequential patterns
    features = []
    for seq in sequences_int:
        ps = PrefixSpan([seq.tolist()])
        patterns = ps.topk(5)
        feat = []
        for p in patterns:
            pat = p[0]
            feat.append(pat[0] if isinstance(pat, list) and pat else pat)
        while len(feat) < 5:
            feat.append(0)
        features.append(feat)

    features = np.array(features)
    features_scaled = StandardScaler().fit_transform(features)

    # Avoid degenerate cases
    if len(np.unique(features_scaled, axis=0)) < 2:
        print("Not enough unique sequences for clustering metrics")
        labels = np.zeros(len(features_scaled))  # fallback
        metrics = {}
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(features_scaled)
        metrics = evaluate_clustering(features_scaled, labels, true_labels)
        save_metrics(metrics, "prefixspan_kmeans")

    save_cluster_labels(labels, "prefixspan_kmeans")
    return labels, metrics


# -------------------- 3. HMM Clustering --------------------
def hmm_clustering(df, window=10, n_states=3, n_clusters=3, true_labels=None):
    sequences = build_sequences(df, window)

    # Train HMM per sequence
    hmms = []
    for seq in sequences:
        model = GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=100
        )
        model.fit(seq)
        # Avoid zero probabilities in transitions
        model.transmat_ = np.where(model.transmat_ == 0, 1e-6, model.transmat_)
        model.transmat_ = model.transmat_ / model.transmat_.sum(
            axis=1, keepdims=True
        )
        hmms.append(model)

    # Distance matrix based on likelihood
    distances = np.zeros((len(sequences), len(sequences)))
    for i, hi in enumerate(hmms):
        for j, seq in enumerate(sequences):
            distances[i, j] = -hi.score(seq)

    # Cluster on distance profiles
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(distances)

    save_cluster_labels(labels, "hmm")
    metrics = evaluate_clustering(sequences, labels, true_labels)
    save_metrics(metrics, "hmm")

    return labels, metrics


# -------------------- 4. DBA-kMeans --------------------
def dba_kmeans_clustering(df, window=10, n_clusters=3, true_labels=None):
    if not TSLEARN_AVAILABLE:
        raise RuntimeError("tslearn not installed. Run: pip install tslearn")

    sequences = build_sequences(df, window)
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_init=5, random_state=42)
    labels = model.fit_predict(sequences)
    centroids = model.cluster_centers_
    save_cluster_labels(labels, "dba_kmeans")
    metrics = evaluate_clustering(sequences, labels, true_labels)
    save_metrics(metrics, "dba_kmeans")

    # Visualization
    plot_cluster_representatives(sequences, labels, centroids, model_name="dba_kmeans")

    return labels, metrics


# -------------------- 5. Sequence Autoencoder + Latent Space Clustering --------------------
def autoencoder_clustering(
    df, window=10, n_clusters=3, latent_dim=10, epochs=20, batch_size=16, true_labels=None
):
    if not KERAS_AVAILABLE:
        raise RuntimeError("TensorFlow/Keras not installed. Run: pip install tensorflow")

    sequences = build_sequences(df, window)
    n_samples, timesteps, n_features = sequences.shape
    sequences_scaled = sequences.astype("float32")

    # Autoencoder
    input_seq = Input(shape=(timesteps, n_features))
    encoded = LSTM(latent_dim)(input_seq)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(n_features, return_sequences=True)(decoded)
    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        sequences_scaled, sequences_scaled, epochs=epochs, batch_size=batch_size, verbose=0
    )

    # Latent vectors
    encoder = Model(input_seq, encoded)
    latent_vectors = encoder.predict(sequences_scaled)

    # Cluster in latent space
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(latent_vectors)

    save_cluster_labels(labels, "autoencoder")
    metrics = evaluate_clustering(sequences_scaled, labels, true_labels)
    save_metrics(metrics, "autoencoder")

    return labels, metrics
