"""
Temperature Clustering-Guided Local Outlier Detection (T-KMeans-LOF)
Implementation of the outlier detection module for the hybrid intelligence framework
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class TKMeansLOF:
    """
    Temperature Clustering-based Local Outlier Factor
    
    This method first clusters data by temperature, then applies LOF 
    within each cluster to identify outliers while preserving 
    temperature-dependent characteristics.
    """
    
    def __init__(self, n_clusters='auto', contamination=0.20, 
                 n_neighbors=3, temperature_col='T/°C'):
        """
        Parameters:
        -----------
        n_clusters : int or 'auto'
            Number of temperature clusters. If 'auto', determined by silhouette score
        contamination : float
            Expected proportion of outliers in the dataset
        n_neighbors : int
            Number of neighbors for LOF algorithm
        temperature_col : str
            Name of temperature column in the dataset
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.temperature_col = temperature_col
        self.kmeans = None
        self.optimal_clusters = None
        
    def _determine_optimal_clusters(self, X_temp, max_clusters=24):
        """
        Determine optimal number of clusters using silhouette score
        
        Parameters:
        -----------
        X_temp : array-like
            Temperature values
        max_clusters : int
            Maximum number of clusters to test
        """
        silhouette_scores = []
        range_n_clusters = range(2, min(max_clusters, len(X_temp)))
        
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_temp.reshape(-1, 1))
            silhouette_avg = silhouette_score(X_temp.reshape(-1, 1), cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find optimal number of clusters
        self.optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
        return self.optimal_clusters
    
    def fit_transform(self, data):
        """
        Fit the model and transform the data by removing outliers
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data containing temperature and composition columns
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data with outliers removed
        """
        df = data.copy()
        
        # Extract temperature values
        X_temp = df[self.temperature_col].values
        
        # Determine optimal clusters if set to 'auto'
        if self.n_clusters == 'auto':
            self.n_clusters = self._determine_optimal_clusters(X_temp)
        
        # Perform temperature clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df['Temp_Cluster'] = self.kmeans.fit_predict(X_temp.reshape(-1, 1))
        
        # Initialize arrays for LOF scores and anomaly labels
        lof_scores = np.zeros(len(df))
        anomaly_labels = np.ones(len(df))
        
        # Apply LOF within each temperature cluster
        composition_cols = [col for col in df.columns 
                          if col not in [self.temperature_col, 'Temp_Cluster']]
        
        for cluster in df['Temp_Cluster'].unique():
            cluster_mask = df['Temp_Cluster'] == cluster
            cluster_data = df.loc[cluster_mask, composition_cols].values
            
            # Skip clusters with insufficient data
            if len(cluster_data) <= self.n_neighbors:
                continue
                
            # Apply LOF
            clf = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(cluster_data)-1),
                contamination=self.contamination
            )
            y_pred = clf.fit_predict(cluster_data)
            lof_cluster_scores = -clf.negative_outlier_factor_
            
            # Update global arrays
            lof_scores[cluster_mask] = lof_cluster_scores
            anomaly_labels[cluster_mask] = y_pred
        
        # Add scores to dataframe
        df['LOF_Score'] = lof_scores
        df['Anomaly'] = anomaly_labels
        
        # Return cleaned data (remove outliers)
        cleaned_data = df[df['Anomaly'] == 1].drop(
            columns=['LOF_Score', 'Anomaly', 'Temp_Cluster']
        )
        
        # Print summary statistics
        n_outliers = np.sum(anomaly_labels == -1)
        print(f"Total outliers detected: {n_outliers} ({n_outliers/len(df)*100:.2f}%)")
        print(f"Cleaned dataset size: {len(cleaned_data)}")
        
        return cleaned_data