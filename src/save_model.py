import joblib
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_and_clean_data
from feature_engineering import build_customer_intelligence_dataset

def export_model():
    print("Loading data to train the final production model...")
    df_clean = load_and_clean_data(r"C:\Users\Alish\OneDrive\Desktop\Mawdoo3 Internship\customer-intelligence-ml\Data\raw_data.xlsx")
    ml_data = build_customer_intelligence_dataset(df_clean)
    
    features = [
        'Recency', 'Frequency', 'Monetary_Value', 
        'Total_Purchased_Quantity', 'Unique_Products_Purchased', 'Average_Order_Value'
    ]
    X = ml_data[features]
    y = ml_data['Target_Next_30_Days']
    
    print("Training final Logistic Regression model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Save the trained model to a file
    joblib.dump(model, 'production_model.pkl')
    print("Success! Model saved as 'production_model.pkl'")

if __name__ == "__main__":
    export_model()