

# Earthquake Data Analysis: Classification & Clustering

This project performs **earthquake data analysis** using both **sequential** and **spatial** approaches, combining classical ML, deep learning, and time-series clustering methods.


## Project Structure

```

earthquake1/
│
├── data/
│   └── earthquakes_last30days.csv   # Earthquake dataset
│
├── results/                        # Generated metrics & cluster labels
│   └── sequential_clustering/
│
├── src/
│   ├── sequential_classification.py
│   ├── sequential_clustering.py
│   ├── spatial_classification.py
│   └── spatial_clustering.py
│
├── main.py                         # Main script to run classifiers/clustering
├── requirements.txt                # Python dependencies
└── README.md                        # Project documentation

````

---

## Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
````

**Key packages:**

* `pandas`, `numpy`, `scikit-learn`, `scipy`
* `tslearn` (DTW & DBA-kMeans)
* `prefixspan` (Sequential pattern mining)
* `hmmlearn` (Hidden Markov Models)
* `tensorflow` (LSTM Autoencoder)
* `sklearn-crfsuite` (Conditional Random Fields)
* `minisom` (Self-Organizing Maps)

---

## Data Format

The CSV dataset should include:

| Column    | Type     | Description               |
| --------- | -------- | ------------------------- |
| time      | datetime | Event timestamp           |
| latitude  | float    | Latitude of earthquake    |
| longitude | float    | Longitude of earthquake   |
| magnitude | float    | Magnitude of earthquake   |
| depth     | float    | Depth of earthquake in km |

---

## Sequential Classification

**Models implemented:**

* Random Forest
* Hidden Markov Model (HMM)
* Conditional Random Field (CRF)
* LSTM (RNN)
* k-NN with DTW

**Run:**

```bash
python main.py   # Sequential classification section
```

---

## Sequential Clustering

**Algorithms implemented:**

* DTW + k-Means
* PrefixSpan + k-Means
* HMM Clustering
* DBA-kMeans
* Sequence Autoencoder + KMeans

**Features:**

* Sliding-window sequence extraction
* Cluster metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn Index
* Cluster representatives visualization for DTW/DBA-kMeans

**Run:**

```bash
python main.py   # Sequential clustering section
```

---

## Spatial Classification

**Algorithms implemented:**

* K-Nearest Neighbors (KNN)
* Decision Tree / Random Forest
* Support Vector Machine (SVM)
* Logistic Regression

**Target classes:**

* `0` = minor earthquake (magnitude < 4)
* `1` = major earthquake (magnitude >= 4)

**Run:**

```bash
python main.py   # Spatial classification section
```

---

## Spatial Clustering

**Algorithms implemented:**

* DBSCAN
* OPTICS
* K-Means
* STING
* Self-Organizing Map (SOM, optional)

**Run:**

```bash
python main.py   # Spatial clustering section
```

---

## Outputs

* **Cluster labels:** Saved in `results/` as CSVs (`*_clusters.csv`)
* **Clustering metrics:** Saved in `results/` (`*_metrics.csv`)
* **Representative sequences (for DTW/DBA-kMeans):** Plot of average sequences per cluster

---

## Notes

* Comment out sections in `main.py` if some dependencies are not installed.
* Ensure **Python >= 3.10** for compatibility.
* For reproducibility, consider using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## Authors

* [Your Name / Team]
* [University / Organization]

---

## License

This project is licensed under [MIT License](LICENSE).

```

---

If you want, I can also **rewrite `main.py` into a single modular version with sections for sequential classification, sequential clustering, spatial classification, and spatial clustering**, so your README commands match exactly.  

Do you want me to do that next?
```
