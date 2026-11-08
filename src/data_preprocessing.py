import pandas as pd

def load_and_clean_data(filepath):
    """Loads raw Excel data and performs initial cleaning."""
    print("Loading raw data (this may take a minute)...")
    all_sheets = pd.read_excel(filepath, sheet_name=None)
    df = pd.concat(all_sheets.values(), ignore_index=True)
    
    print("Cleaning missing values and cancellations...")
    df_clean = df.dropna(subset=['Customer ID']).copy()
    df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
    df_clean = df_clean.drop_duplicates()
    
    # Create the baseline financial column
    df_clean['Total_Price'] = df_clean['Quantity'] * df_clean['Price']
    
    return df_clean