# # main.py sequential classification
# import os
# import pandas as pd
# from src.sequential_classification import (
#     sequence_random_forest,
#     hmm_classifier,
#     crf_classifier,
#     lstm_classifier,
#     dtw_knn_classifier,
#     HMM_AVAILABLE,
#     CRF_AVAILABLE,
#     KERAS_AVAILABLE,
#     TSLEARN_AVAILABLE
# )

# DATA_PATH = "data/earthquakes_last30days.csv"

# def load_data(path=DATA_PATH):
#     """Load earthquake dataset from CSV and clean."""
#     df = pd.read_csv(path)
#     if "time" in df.columns:
#         df["time"] = pd.to_datetime(df["time"], errors="coerce")
#     return df.dropna(subset=["magnitude"])

# def run_classifier(classifier_fn, df, name, **kwargs):
#     """Run a classifier safely and print status."""
#     try:
#         print(f"\n====================== {name} ======================")
#         model = classifier_fn(df, **kwargs)
#         print(f"{name} completed successfully.")
#         return model
#     except Exception as e:
#         print(f"{name} skipped due to error: {e}")
#         return None

# def main():
#     df = load_data()
#     completed = []

#     # Random Forest
#     rf_model = run_classifier(sequence_random_forest, df, "Random Forest")
#     if rf_model: completed.append("Random Forest")

#     # HMM
#     if HMM_AVAILABLE:
#         hmm_model = run_classifier(hmm_classifier, df, "Hidden Markov Model")
#         if hmm_model: completed.append("Hidden Markov Model")
#     else:
#         print("Skipping HMM: hmmlearn not installed")

#     # CRF
#     if CRF_AVAILABLE:
#         crf_model = run_classifier(crf_classifier, df, "Conditional Random Field")
#         if crf_model: completed.append("Conditional Random Field")
#     else:
#         print("Skipping CRF: sklearn-crfsuite not installed")

#     # LSTM
#     if KERAS_AVAILABLE:
#         lstm_model = run_classifier(lstm_classifier, df, "LSTM (RNN)", epochs=10)
#         if lstm_model: completed.append("LSTM (RNN)")
#     else:
#         print("Skipping LSTM: TensorFlow/Keras not installed")

#     # DTW k-NN
#     if TSLEARN_AVAILABLE:
#         dtw_model = run_classifier(dtw_knn_classifier, df, "k-NN with DTW")
#         if dtw_model: completed.append("k-NN with DTW")
#     else:
#         print("Skipping DTW k-NN: tslearn not installed")

#     print("\n====================== Summary ======================")
#     print("Completed classifiers:", ", ".join(completed))

# if __name__ == "_main_":
#     main()
##---------------------------------------------------------------------------------------------------------------

# # main.py sequential clustering
# import pandas as pd
# from src.sequential_clustering import (
#     dtw_kmeans_clustering,
#     prefixspan_kmeans_clustering,
#     hmm_clustering,
#     dba_kmeans_clustering,
#     autoencoder_clustering,
#     TSLEARN_AVAILABLE,
#     PREFIXSPAN_AVAILABLE,
#     HMM_AVAILABLE,
#     KERAS_AVAILABLE
# )

# def load_data(path="data/earthquakes_last30days.csv"):
#     """Load earthquake dataset from CSV."""
#     df = pd.read_csv(path)
#     if "time" in df.columns:
#         df["time"] = pd.to_datetime(df["time"], errors="coerce")
#     return df.dropna(subset=["magnitude"])

# def main():
#     df = load_data()

#     print("\n====================== DTW + k-Means ======================")
#     if TSLEARN_AVAILABLE:
#         dtw_kmeans_clustering(df)
#     else:
#         print("Skipping DTW k-Means: tslearn not available")

#     print("\n====================== PrefixSpan + k-Means ======================")
#     if PREFIXSPAN_AVAILABLE:
#         prefixspan_kmeans_clustering(df)
#     else:
#         print("Skipping PrefixSpan k-Means: prefixspan not available")

#     print("\n====================== HMM Clustering ======================")
#     if HMM_AVAILABLE:
#         hmm_clustering(df)
#     else:
#         print("Skipping HMM Clustering: hmmlearn not available")

#     print("\n====================== DBA-kMeans ======================")
#     if TSLEARN_AVAILABLE:
#         dba_kmeans_clustering(df)
#     else:
#         print("Skipping DBA-kMeans: tslearn not available")

#     print("\n====================== Sequence Autoencoder + KMeans ======================")
#     if KERAS_AVAILABLE:
#         autoencoder_clustering(df)
#     else:
#         print("Skipping Autoencoder Clustering: TensorFlow/Keras not available")

# if __name__ == "_main_":
#     main()
##---------------------------------------------------------------------------------------------------------------------------------


# #main.py spatial classification
# # main.py
# import pandas as pd
# from src.spatial_classification import (
#     knn_classification,
#     decision_tree_classification,
#     svm_classification,
#     logistic_regression_classification
# )

# def load_data(path="data/earthquakes_last30days.csv"):
#     df = pd.read_csv(path)
#     df = df.dropna(subset=["latitude", "longitude", "magnitude"])
#     df["class"] = (df["magnitude"] >= 4.0).astype(int)  # 0 = minor, 1 = major
#     return df

# def main():
#     df = load_data()
#     features = ["latitude", "longitude", "magnitude", "depth"]
    
#     print("\n================ K-Nearest Neighbors ================")
#     knn_classification(df.copy(), features)
    
#     print("\n================ Random Forest ======================")
#     decision_tree_classification(df.copy(), features)
    
#     print("\n================ Support Vector Machine =============")
#     svm_classification(df.copy(), features)
    
#     print("\n================ Logistic Regression ===============")
#     logistic_regression_classification(df.copy(), features)

# if __name__ == "_main_":
#     main()
##-----------------------------------------------------------------------------------------------------------------------------------------------


# # main.py spatial_clustering
# import pandas as pd
# from src.spatial_clustering import (
#     dbscan_clustering,
#     optics_clustering,
#     kmeans_clustering,
#     sting_clustering,
#     som_clustering,
#     SOM_AVAILABLE
# )

# def load_data(path="data/earthquakes_last30days.csv"):
#     """Load earthquake dataset from CSV."""
#     df = pd.read_csv(path)
#     if "time" in df.columns:
#         df["time"] = pd.to_datetime(df["time"], errors="coerce")
#     df = df.dropna(subset=["latitude", "longitude"])
#     # Add a dummy 'class' column if needed for evaluation
#     if "class" not in df.columns:
#         df["class"] = 0
#     return df

# def main():
#     df = load_data()

#     print("\n====================== DBSCAN ======================")
#     df_dbscan, metrics_dbscan = dbscan_clustering(df)
#     print("DBSCAN metrics:", metrics_dbscan)

#     print("\n====================== OPTICS ======================")
#     df_optics, metrics_optics = optics_clustering(df)
#     print("OPTICS metrics:", metrics_optics)

#     print("\n====================== K-Means ======================")
#     df_kmeans, metrics_kmeans = kmeans_clustering(df)
#     print("K-Means metrics:", metrics_kmeans)

#     print("\n====================== STING ======================")
#     df_sting, metrics_sting = sting_clustering(df)
#     print("STING metrics:", metrics_sting)

#     print("\n====================== SOM ======================")
#     if SOM_AVAILABLE:
#         df_som, metrics_som = som_clustering(df)
#         print("SOM metrics:", metrics_som)
#     else:
#         print("Skipping SOM: minisom not installed")

# if _name_ == "_main_":
#     main()


##----------------------------------------------------------------------------------------------------------------------------------------------------
