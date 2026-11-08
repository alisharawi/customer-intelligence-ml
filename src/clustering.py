from data_preprocessing import load_and_clean_data
from feature_engineering import build_customer_intelligence_dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def segment_customers():
    # 1. Load and prepare the data using our existing pipeline
    print("Loading data and building features for segmentation...")
    filepath = r"C:\Users\Alish\OneDrive\Desktop\Mawdoo3 Internship\customer-intelligence-ml\Data\raw_data.xlsx"  # Adjust path if needed
    df_clean = load_and_clean_data(filepath)
    customer_data = build_customer_intelligence_dataset(df_clean)
    
    # We only need the core RFM metrics for clustering
    rfm = customer_data[['Customer ID', 'Recency', 'Frequency', 'Monetary_Value']].copy()
    
    # 2. Handle 0 or negative values before applying Log Transformation
    # Log transformation fails on numbers <= 0, so we add a small constant (1)
    print("Scaling data to handle financial outliers...")
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary_Value']].apply(lambda x: np.log(x + 1))
    
    # 3. Standardize the data so Recency (days) and Monetary (pounds) are on the same scale
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # 4. Apply K-Means Clustering (We will start with 4 distinct customer groups)
    print("Training K-Means Clustering model...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 5. Analyze the results by calculating the average stats for each cluster
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary_Value': ['mean', 'count']
    }).round(2)
    
    print("\n--- Customer Segmentation Summary ---")
    print(cluster_summary)
    
    # 6. Visualize the Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Recency', y='Monetary_Value', hue='Cluster', 
        data=rfm, palette='viridis', alpha=0.6
    )
    plt.title('Customer Segments: Recency vs. Monetary Value', fontsize=14)
    plt.xlabel('Recency (Days since last purchase)')
    plt.ylabel('Monetary Value (£)')
    plt.yscale('log') # Use log scale for visibility
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    segment_customers()