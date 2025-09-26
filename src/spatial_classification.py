# src/spatial_classification.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
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
    pd.DataFrame([metrics]).to_csv(path, index=False)


def save_predictions(df, model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_predictions.csv")
    df.to_csv(path, index=False)


def _build_legend_html(class_labels, colors):
    # returns small HTML block to add to folium map with legend entries
    html = """
    <div style="
      position: fixed;
      bottom: 50px;
      left: 50px;
      width: 160px;
      max-height: 300px;
      overflow:auto;
      padding:8px;
      background-color: white;
      border:2px solid grey;
      z-index:9999;
      font-size:14px;
    ">
    &nbsp;<b>Legend</b><br>
    """
    for i, c in enumerate(class_labels):
        color = colors[i % len(colors)]
        html += f"&nbsp;<i style='background:{color};width:10px;height:10px;display:inline-block;margin-right:6px'></i>Class {c}<br>"
    html += "</div>"
    return html


def plot_map(df, lat_col="latitude", lon_col="longitude", class_col="predicted", model_name="model"):
    """Save a folium map showing predicted classes (uses df rows provided)."""
    if df.empty:
        return
    center = [float(df[lat_col].mean()), float(df[lon_col].mean())]
    m = folium.Map(location=center, zoom_start=5)
    colors = ["blue", "red", "green", "orange", "purple", "darkred", "cadetblue", "darkgreen"]

    class_labels = sorted(pd.Series(df[class_col]).unique())

    for _, row in df.iterrows():
        try:
            color = colors[int(row[class_col]) % len(colors)]
        except Exception:
            color = colors[0]
        folium.CircleMarker(
            location=[float(row[lat_col]), float(row[lon_col])],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    # add legend
    legend_html = _build_legend_html(class_labels, colors)
    m.get_root().html.add_child(folium.Element(legend_html))

    map_path = os.path.join(RESULTS_DIR, f"{model_name}_map.html")
    m.save(map_path)


# -------------------- Evaluation Function --------------------
def evaluate_classification(y_true, y_pred):
    metrics = {}
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Core Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Extended Metrics
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
    _for = fn / (fn + tn) if (fn + tn) > 0 else 0  # False Omission Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    plr = (recall / (1 - specificity)) if (1 - specificity) > 0 else float("inf")
    nlr = ((1 - recall) / specificity) if specificity > 0 else float("inf")
    prevalence = (tp + fn) / (tp + tn + fp + fn)
    balanced_acc = (recall + specificity) / 2

    # Save in dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "npv": npv,
        "fdr": fdr,
        "for": _for,
        "fnr": fnr,
        "fpr": fpr,
        "plr": plr,
        "nlr": nlr,
        "prevalence": prevalence,
        "balanced_accuracy": balanced_acc
    }
    
    return metrics

# -------------------- Classifiers (all use train/test) --------------------
def knn_classification(df, features, target="class", n_neighbors=5, test_size=0.2, random_state=42):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # create test dataframe with predictions and original lat/lon if available
    df_test = df.iloc[y_test.index].copy() if hasattr(y_test, "index") else df.iloc[X_test.shape[0]*[0]].copy()
    df_test = df_test.reset_index(drop=True)
    df_test["predicted"] = y_pred

    metrics = evaluate_classification(y_test, y_pred)
    save_predictions(df_test, "knn")
    save_metrics(metrics, "knn")
    plot_map(df_test, class_col="predicted", model_name="knn")
    return df_test, metrics


def decision_tree_classification(df, features, target="class", test_size=0.2, random_state=42):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_test = df.iloc[y_test.index].copy() if hasattr(y_test, "index") else df.iloc[X_test.shape[0]*[0]].copy()
    df_test = df_test.reset_index(drop=True)
    df_test["predicted"] = y_pred

    metrics = evaluate_classification(y_test, y_pred)
    save_predictions(df_test, "decision_tree")
    save_metrics(metrics, "decision_tree")
    plot_map(df_test, class_col="predicted", model_name="decision_tree")
    return df_test, metrics


def random_forest_classification(df, features, target="class", test_size=0.2, random_state=42):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_test = df.iloc[y_test.index].copy() if hasattr(y_test, "index") else df.iloc[X_test.shape[0]*[0]].copy()
    df_test = df_test.reset_index(drop=True)
    df_test["predicted"] = y_pred

    metrics = evaluate_classification(y_test, y_pred)
    save_predictions(df_test, "random_forest")
    save_metrics(metrics, "random_forest")
    plot_map(df_test, class_col="predicted", model_name="random_forest")
    return df_test, metrics


def svm_classification(df, features, target="class", test_size=0.2, random_state=42):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    model = SVC(kernel="rbf", probability=True, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_test = df.iloc[y_test.index].copy() if hasattr(y_test, "index") else df.iloc[X_test.shape[0]*[0]].copy()
    df_test = df_test.reset_index(drop=True)
    df_test["predicted"] = y_pred

    metrics = evaluate_classification(y_test, y_pred)
    save_predictions(df_test, "svm")
    save_metrics(metrics, "svm")
    plot_map(df_test, class_col="predicted", model_name="svm")
    return df_test, metrics


def logistic_regression_classification(df, features, target="class", test_size=0.2, random_state=42):
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    model = LogisticRegression(max_iter=500, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_test = df.iloc[y_test.index].copy() if hasattr(y_test, "index") else df.iloc[X_test.shape[0]*[0]].copy()
    df_test = df_test.reset_index(drop=True)
    df_test["predicted"] = y_pred

    metrics = evaluate_classification(y_test, y_pred)
    save_predictions(df_test, "logistic_regression")
    save_metrics(metrics, "logistic_regression")
    plot_map(df_test, class_col="predicted", model_name="logistic_regression")
    return df_test, metrics
