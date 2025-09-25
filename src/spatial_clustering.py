# src/spatial_clustering.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    fowlkes_mallows_score, v_measure_score
)
import folium

# Optional SOM import
try:
    from minisom import MiniSom
    SOM_AVAILABLE = True
except Exception:
    SOM_AVAILABLE = False

RESULTS_DIR = "results/spatial_clustering"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------- Helpers --------------------
def save_metrics(metrics, model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(path, index=False)

def save_cluster_labels(labels, model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_clusters.csv")
    pd.DataFrame({"sequence_id": range(len(labels)), "cluster": labels}).to_csv(path, index=False)

def plot_map(df, lat_col="latitude", lon_col="longitude", cluster_col="cluster", model_name="model"):
    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=5)
    colors = ["blue", "red", "green", "orange", "purple", "darkred", "lightblue", "cadetblue"]
    cluster_labels = sorted(df[cluster_col].unique())
    
    for _, row in df.iterrows():
        color = colors[int(row[cluster_col]) % len(colors)] if row[cluster_col] != -1 else "black"  # noise in black
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.6
        ).add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 150px;
    height: {30*len(cluster_labels)+10}px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;">
    &nbsp;<b>Clusters</b><br>
    """
    for i, c in enumerate(cluster_labels):
        color = colors[i % len(colors)] if c != -1 else "black"
        legend_html += f"&nbsp;<i style='background:{color};width:10px;height:10px;display:inline-block'></i>&nbsp;Cluster {c}<br>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    map_path = os.path.join(RESULTS_DIR, f"{model_name}_map.html")
    m.save(map_path)

def evaluate_clustering(X, labels, y_true=None):
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics["silhouette_score"] = silhouette_score(X, labels)
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
    else:
        metrics["silhouette_score"] = np.nan
        metrics["calinski_harabasz"] = np.nan
        metrics["davies_bouldin"] = np.nan
    if y_true is not None and len(np.unique(labels)) > 1:
        metrics["fowlkes_mallows"] = fowlkes_mallows_score(y_true, labels)
        metrics["v_measure"] = v_measure_score(y_true, labels)
    else:
        metrics["fowlkes_mallows"] = np.nan
        metrics["v_measure"] = np.nan
    return metrics

def preprocess_coordinates(df, lat_col="latitude", lon_col="longitude"):
    X = df[[lat_col, lon_col]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# -------------------- 1. DBSCAN --------------------
def dbscan_clustering(df, eps=0.5, min_samples=5):
    X = preprocess_coordinates(df)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    df["cluster"] = labels
    metrics = evaluate_clustering(X, labels)
    save_cluster_labels(labels, "dbscan")
    save_metrics(metrics, "dbscan")
    plot_map(df, cluster_col="cluster", model_name="dbscan")
    return df, metrics

# -------------------- 2. OPTICS --------------------
def optics_clustering(df, min_samples=5, xi=0.05, min_cluster_size=0.05):
    X = preprocess_coordinates(df)
    model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    df["cluster"] = labels
    metrics = evaluate_clustering(X, labels)
    save_cluster_labels(labels, "optics")
    save_metrics(metrics, "optics")
    plot_map(df, cluster_col="cluster", model_name="optics")
    return df, metrics

# -------------------- 3. K-Means --------------------
def kmeans_clustering(df, n_clusters=3):
    X = preprocess_coordinates(df)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    df["cluster"] = labels
    metrics = evaluate_clustering(X, labels)
    save_cluster_labels(labels, "kmeans")
    save_metrics(metrics, "kmeans")
    plot_map(df, cluster_col="cluster", model_name="kmeans")
    return df, metrics

# -------------------- 4. STING (grid-based clustering) --------------------
def sting_clustering(df, grid_size=(10,10)):
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    n_cells_y, n_cells_x = grid_size
    cell_height = (lat_max - lat_min) / n_cells_y
    cell_width = (lon_max - lon_min) / n_cells_x
    labels = np.zeros(len(df), dtype=int)
    
    for i, row in df.iterrows():
        y_idx = int((row["latitude"] - lat_min) / cell_height)
        x_idx = int((row["longitude"] - lon_min) / cell_width)
        labels[i] = y_idx * n_cells_x + x_idx
    
    df["cluster"] = labels
    X = preprocess_coordinates(df)
    metrics = evaluate_clustering(X, labels)
    save_cluster_labels(labels, "sting")
    save_metrics(metrics, "sting")
    plot_map(df, cluster_col="cluster", model_name="sting")
    return df, metrics

# -------------------- 5. Self-Organizing Map (SOM) --------------------
def som_clustering(df, x_dim=5, y_dim=5, num_iterations=100):
    if not SOM_AVAILABLE:
        raise RuntimeError("minisom not installed. Run: pip install minisom")
    X = preprocess_coordinates(df)
    som = MiniSom(x_dim, y_dim, X.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, num_iterations)
    
    labels = np.array([np.ravel_multi_index(som.winner(x), (x_dim, y_dim)) for x in X])
    df["cluster"] = labels
    metrics = evaluate_clustering(X, labels)
    save_cluster_labels(labels, "som")
    save_metrics(metrics, "som")
    plot_map(df, cluster_col="cluster", model_name="som")
    return df, metrics
