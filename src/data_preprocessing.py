import pandas as pd

all_sheets = pd.read_excel(r"C:\Users\Alish\OneDrive\Desktop\Mawdoo3 Internship\customer-intelligence-ml\Data\raw_data.xlsx"
, sheet_name=None)

# Combine all the DataFrames from the dictionary into one single DataFrame
df = pd.concat(all_sheets.values(), ignore_index=True)
df_clean = df.dropna(subset=['Customer ID']).copy()
df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
print(f"original rows {len(df)}")
print(f"cleaned rows {len(df_clean)}")
df_clean.describe()
#df_clean.info()
#no_id.head(10)
duplicate_rows = df_clean.duplicated().sum()
print(duplicate_rows)
df_clean = df_clean.drop_duplicates()
df_clean.info()
df_clean['Total Price'] = df_clean['Quantity'] * df_clean['Price']
top_10_customers = df_clean.groupby('Customer ID')['Total Price'].sum().nlargest(10)
print(top_10_customers)
df_clean.head(10)

