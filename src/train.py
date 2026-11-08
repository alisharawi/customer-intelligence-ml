from data_preprocessing import load_and_clean_data
from feature_engineering import build_customer_intelligence_dataset

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r'C:\Users\Alish\OneDrive\Desktop\Mawdoo3 Internship\customer-intelligence-ml\Data\raw_data.xlsx'

# Combine all the DataFrames from the dictionary into one single DataFrame
df_clean = load_and_clean_data(file_path)
ml_data = build_customer_intelligence_dataset(df_clean)

features = [
    'Recency', 'Frequency', 'Monetary_Value', 'Total_Purchased_Quantity', 'Unique_Products_Purchased', 'Average_Order_Value'
]

x = ml_data[features]
y = ml_data['Target_Next_30_Days']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

RFC_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
RFC_model.fit(X_train, y_train)

RFC_predictions = RFC_model.predict(X_test)
print(classification_report(y_test, RFC_predictions))

plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, RFC_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Purchase (0)', 'Purchase (1)'], 
            yticklabels=['No Purchase (0)', 'Purchase (1)'])
plt.title('Prediction Accuracy on Test Data', fontsize=14)
plt.xlabel('What the Model Predicted', fontsize=12)
plt.ylabel('What Actually Happened', fontsize=12)
plt.tight_layout()
plt.show()