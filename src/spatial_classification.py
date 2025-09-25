# src/spatial_classification.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score, v_measure_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import folium

RESULTS_DIR = "results/spatial_classification"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------- Helpers --------------------
def save_metrics(metrics, model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.csv")
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)

def save_predictions(df, model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_predictions.csv")
    df.to_csv(path, index=False)

def plot_map(df, lat_col="latitude", lon_col="longitude", class_col="predicted", model_name="model"):
    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=5)
    colors = ["blue", "red", "green", "orange", "purple"]
    class_labels = sorted(df[class_col].unique())

    for _, row in df.iterrows():
        color = colors[int(row[class_col]) % len(colors)]
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    # Add legend
    legend_html = """
     <div style="
     position: fixed;
     bottom: 50px;
     left: 50px;
     width: 120px;
     height: {}px;
     background-color: white;
     border:2px solid grey;
     z-index:9999;
     font-size:14px;
     ">
     &nbsp;<b>Legend</b><br>
    """.format(30 * len(class_labels) + 10)

    for i, c in enumerate(class_labels):
        legend_html += f"&nbsp;<i style='background:{colors[i % len(colors)]};width:10px;height:10px;display:inline-block'></i>&nbsp;Class {c}<br>"
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
    map_path = os.path.join(RESULTS_DIR, f"{model_name}_map.html")
    m.save(map_path)

def evaluate_clustering(X, labels, y_true=None):
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
    else:
        metrics["calinski_harabasz"] = np.nan
        metrics["davies_bouldin"] = np.nan
    if y_true is not None and len(np.unique(labels)) > 1:
        metrics["fowlkes_mallows"] = fowlkes_mallows_score(y_true, labels)
        metrics["v_measure"] = v_measure_score(y_true, labels)
    else:
        metrics["fowlkes_mallows"] = np.nan
        metrics["v_measure"] = np.nan
    return metrics

def rasterize_earthquakes(df, grid_size=(50,50), features=["magnitude", "depth"]):
    """Convert earthquake points into a raster grid."""
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    
    raster = np.zeros((grid_size[0], grid_size[1], len(features)))
    
    for i, feat in enumerate(features):
        vals = df[feat].fillna(0).values
        lat_idx = np.floor((df["latitude"] - lat_min) / (lat_max - lat_min) * (grid_size[0]-1)).astype(int)
        lon_idx = np.floor((df["longitude"] - lon_min) / (lon_max - lon_min) * (grid_size[1]-1)).astype(int)
        for y, x, v in zip(lat_idx, lon_idx, vals):
            raster[y, x, i] += v  # sum values in same cell
    return raster, (lat_min, lat_max, lon_min, lon_max)
# -------------------- Classifiers --------------------
def knn_classification(df, features, target="class", n_neighbors=5):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_scaled)
    
    df["predicted"] = y_pred
    metrics = evaluate_clustering(X_scaled, y_pred, y_true=y)
    save_predictions(df, "knn")
    save_metrics(metrics, "knn")
    plot_map(df, class_col="predicted", model_name="knn")
    return df, metrics

def decision_tree_classification(df, features, target="class"):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    
    df["predicted"] = y_pred
    metrics = evaluate_clustering(X_scaled, y_pred, y_true=y)
    save_predictions(df, "random_forest")
    save_metrics(metrics, "random_forest")
    plot_map(df, class_col="predicted", model_name="random_forest")
    return df, metrics

def svm_classification(df, features, target="class"):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    
    df["predicted"] = y_pred
    metrics = evaluate_clustering(X_scaled, y_pred, y_true=y)
    save_predictions(df, "svm")
    save_metrics(metrics, "svm")
    plot_map(df, class_col="predicted", model_name="svm")
    return df, metrics

def logistic_regression_classification(df, features, target="class"):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    
    df["predicted"] = y_pred
    metrics = evaluate_clustering(X_scaled, y_pred, y_true=y)
    save_predictions(df, "logistic_regression")
    save_metrics(metrics, "logistic_regression")
    plot_map(df, class_col="predicted", model_name="logistic_regression")
    return df, metrics

