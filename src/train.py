from data_preprocessing import load_and_clean_data
from feature_engineering import build_customer_intelligence_dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def precision_at_k(y_true, y_prob, k=100):
    """Calculates Precision@K: Out of the top K customers we predict are most likely to buy, how many actually did?"""
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values(by='prob', ascending=False)
    top_k = df.head(k)
    return top_k['true'].mean()

def train_and_compare_models():
    print("Loading data and building features...")
    filepath = r"C:\Users\Alish\OneDrive\Desktop\Mawdoo3 Internship\customer-intelligence-ml\Data\raw_data.xlsx"
    df_clean = load_and_clean_data(filepath)
    ml_data = build_customer_intelligence_dataset(df_clean)
    
    features = [
        'Recency', 'Frequency', 'Monetary_Value', 
        'Total_Purchased_Quantity', 'Unique_Products_Purchased', 'Average_Order_Value'
    ]
    X = ml_data[features]
    y = ml_data['Target_Next_30_Days']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "1. Baseline": DummyClassifier(strategy='prior'),
        "2. Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "3. Decision Tree": DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42),
        "4. Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "5. Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    print("\nEvaluating models across all required metrics...\n")
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        # Get the probability of class 1 (buying) for the advanced metrics
        y_prob = model.predict_proba(X_test)[:, 1] 
        
        # 1. Confusion Matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cm_str = f"TP:{tp} FP:{fp} TN:{tn} FN:{fn}"
        
        # 2-4. Core Classification Metrics
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 5-6. Area Under Curve Metrics (Crucial for imbalanced data)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        # 7. Precision @ K (Top 100 highest-probability customers)
        p_at_k = precision_at_k(y_test, y_prob, k=100)
        
        results.append({
            "Model": name,
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3),
            "ROC-AUC": round(roc_auc, 3),
            "PR-AUC": round(pr_auc, 3),
            "Precision@100": round(p_at_k, 3),
            "Confusion Matrix": cm_str
        })
        
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    train_and_compare_models()