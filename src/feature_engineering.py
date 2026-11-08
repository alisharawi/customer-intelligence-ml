import pandas as pd
import datetime as dt

def build_customer_intelligence_dataset(df, target_window_days=30):
    """
    Phase 2 Reusable Preprocessing Pipeline.
    Cleans data, engineers features, and prevents data leakage via temporal splitting.
    """
    # 1. Clean Data (Requirements 1, 2, 3)
    df_clean = df.dropna(subset=['Customer ID']).copy()
    df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
    df_clean = df_clean.drop_duplicates()
    df_clean['Total_Price'] = df_clean['Quantity'] * df_clean['Price']
    
    # 2. Temporal Split to prevent Data Leakage (Requirement 7)
    max_date = df_clean['InvoiceDate'].max()
    cutoff_date = max_date - pd.Timedelta(days=target_window_days)
    
    feature_window = df_clean[df_clean['InvoiceDate'] <= cutoff_date]
    target_window = df_clean[df_clean['InvoiceDate'] > cutoff_date]
    buyers_in_target = target_window['Customer ID'].unique()
    
    ref_date = cutoff_date + pd.Timedelta(days=1)
    
    # 3. Aggregate Features (Requirements 4 & 5)
    customer_features = feature_window.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
        'Invoice': 'nunique',                                # Total Historical Orders (Frequency)
        'Total_Price': 'sum',                                # Monetary Value
        'Quantity': 'sum',                                   # Total Purchased Quantity
        'StockCode': 'nunique'                               # Number of Unique Products Purchased
    }).reset_index()
    
    # Rename columns to match the project requirements exactly
    customer_features.columns = [
        'Customer ID', 'Recency', 'Frequency', 'Monetary_Value', 
        'Total_Purchased_Quantity', 'Unique_Products_Purchased'
    ]
    
    # Calculate Average Order Value
    customer_features['Average_Order_Value'] = customer_features['Monetary_Value'] / customer_features['Frequency']
    
    # 4. Create the Target Variable for Machine Learning
    customer_features['Target_Next_30_Days'] = customer_features['Customer ID'].isin(buyers_in_target).astype(int)
    
    return customer_features

