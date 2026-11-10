import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Import your existing pipeline functions
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import build_customer_intelligence_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Customer Intelligence Hub", layout="wide")
st.title("📊 Customer Intelligence & Purchase Prediction System")

# --- CACHE DATA & MODELS ---
# @st.cache_data prevents the app from reloading the huge Excel file every time you click a button
@st.cache_data
def get_data():
    df_clean = load_and_clean_data(r"C:\Users\Alish\OneDrive\Desktop\Mawdoo3 Internship\customer-intelligence-ml\Data\raw_data.xlsx")
    ml_data = build_customer_intelligence_dataset(df_clean)
    return df_clean, ml_data

@st.cache_resource
def get_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'production_model.pkl')

    return joblib.load(model_path)

with st.spinner("Loading database and models... (This takes a moment on startup)"):
    df, customers = get_data()
    model = get_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a Dashboard Section:", 
    ["A. Business Analytics", "B. Customer Segmentation", "C. Purchase Prediction", "D. Model Evaluation"]
)

# ==========================================
# A. BUSINESS ANALYTICS
# ==========================================
if page == "A. Business Analytics":
    st.header("📈 Business Analytics & Sales Trends")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{df['Customer ID'].nunique():,}")
    col2.metric("Total Transactions", f"{df['Invoice'].nunique():,}")
    col3.metric("Historical Revenue", f"£{df['Total_Price'].sum():,.2f}")
    
    # Sales Trend Chart
    st.subheader("Monthly Sales Trend")
    monthly_sales = df.set_index('InvoiceDate').resample('ME')['Total_Price'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=monthly_sales, x='InvoiceDate', y='Total_Price', marker="o", ax=ax)
    ax.set_ylabel("Revenue (£)")
    ax.set_xlabel("Date")
    st.pyplot(fig)

# ==========================================
# B. CUSTOMER SEGMENTATION
# ==========================================
elif page == "B. Customer Segmentation":
    st.header("👥 Customer Segmentation (K-Means)")
    st.write("Grouping customers based on their purchasing behavior to identify high-value segments.")
    
    # Run clustering dynamically
    rfm = customers[['Customer ID', 'Recency', 'Frequency', 'Monetary_Value']].copy()
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary_Value']].apply(lambda x: np.log(x + 1))
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Display Summary Table
    st.subheader("Segment Characteristics")
    summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary_Value': ['mean', 'count']
    }).round(1)
    st.dataframe(summary)
    
    # Display Chart
    st.subheader("Customer Value Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary_Value', hue='Cluster', palette='viridis', alpha=0.6, ax=ax)
    ax.set_yscale('log')
    ax.set_title("Recency vs. Monetary Value by Segment")
    st.pyplot(fig)

# ==========================================
# C. PURCHASE PREDICTION
# ==========================================
elif page == "C. Purchase Prediction":
    st.header("🔮 30-Day Purchase Prediction Simulator")
    st.write("Adjust the customer metrics below to predict if they will return to buy in the next 30 days.")
    
    col1, col2 = st.columns(2)
    with col1:
        recency = st.slider("Days Since Last Purchase (Recency)", 0, 365, 15)
        frequency = st.number_input("Total Historical Orders (Frequency)", min_value=1, value=12)
        monetary = st.number_input("Total Amount Spent (£)", min_value=1.0, value=1500.0)
    with col2:
        qty = st.number_input("Total Items Purchased", min_value=1, value=300)
        unique_items = st.number_input("Unique Products Bought", min_value=1, value=45)
        aov = st.number_input("Average Order Value (£)", min_value=1.0, value=125.0)

    if st.button("Predict Future Purchase"):
        # Format for model
        input_features = [[recency, frequency, monetary, qty, unique_items, aov]]
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]
        
        st.divider()
        if prediction == 1:
            st.success(f"### 🎉 Likely to Purchase!")
            st.write(f"**Estimated Probability:** {probability * 100:.1f}%")
        else:
            st.error(f"### ⚠️ Unlikely to Purchase")
            st.write(f"**Estimated Probability:** {probability * 100:.1f}%")

# ==========================================
# D. MODEL EVALUATION
# ==========================================
elif page == "D. Model Evaluation":
    st.header("⚙️ Model Evaluation & Interpretation")
    st.write("Dynamically evaluating the performance of our deployed Logistic Regression model.")
    
    # 1. Prepare Data & Run Live Evaluation
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.dummy import DummyClassifier
    
    features = ['Recency', 'Frequency', 'Monetary_Value', 'Total_Purchased_Quantity', 'Unique_Products_Purchased', 'Average_Order_Value']
    X = customers[features]
    y = customers['Target_Next_30_Days']
    
    # Split the data exactly as we did during the training phase
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Get live predictions from the deployed model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Create a baseline model (guessing the majority class) for comparison
    dummy = DummyClassifier(strategy='prior')
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_prob = dummy.predict_proba(X_test)[:, 1]
    
    # --- MODEL PERFORMANCE COMPARISON & METRICS ---
    st.subheader("1. Model Performance Comparison")
    st.write("Comparing our intelligent Machine Learning model against a baseline that just guesses 'No' for every customer.")
    
    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Deployed ML Model": [
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred, zero_division=0),
            f1_score(y_test, y_pred, zero_division=0),
            roc_auc_score(y_test, y_prob)
        ],
        "Baseline (Guessing No)": [
            precision_score(y_test, dummy_pred, zero_division=0),
            recall_score(y_test, dummy_pred, zero_division=0),
            f1_score(y_test, dummy_pred, zero_division=0),
            roc_auc_score(y_test, dummy_prob)
        ]
    }).set_index("Metric")
    
    # Display the comparison table
    st.dataframe(metrics_df.style.format("{:.3f}"))
    
    # --- CONFUSION MATRIX ---
    st.subheader("2. Confusion Matrix")
    st.write("Visualizing True Positives (successful predictions) vs. False Positives (errors).")
    
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=['Predicted No', 'Predicted Yes'], 
                yticklabels=['Actual No', 'Actual Yes'])
    st.pyplot(fig_cm)
    
    # --- MODEL INTERPRETATION ---
    st.subheader("3. Model Interpretation (Feature Importance)")
    st.write("Which behaviors drive repeat purchases the most?")
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance (Weight)': model.coef_[0]
    }).sort_values(by='Importance (Weight)', ascending=False)
    
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    sns.barplot(data=importance, x='Importance (Weight)', y='Feature', ax=ax_imp, palette='coolwarm')
    st.pyplot(fig_imp)