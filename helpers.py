import pandas as pd
import numpy as np
from extensions import db

# Load scaler and model globally
import joblib
scaler = joblib.load('scaler.pkl')

cat_map = {
    'MerchantCategory': {'Electronics': 0, 'Restaurants': 1, 'Groceries': 2, 'Clothing': 3, 'Entertainment': 4, 'Utilities': 5, 'Travel': 6},
    'TransactionType': {'P2M': 0, 'P2P': 1},
    'TransactionFrequency': {'5/day': 0, '3/day': 1, '1/day': 2},
    'BankName': {'Bank of Baroda': 0, 'ICICI Bank': 1, 'State Bank of India': 2, 'Axis Bank': 3, 'Kotak Mahindra Bank': 4, 'HDFC Bank': 5},
    'UnusualLocation': {'Yes': 1, 'No': 0},
    'UnusualAmount': {'Yes': 1, 'No': 0},
    'NewDevice': {'Yes': 1, 'No': 0}
}

def preprocess_and_encode(df):
    # Map categorical columns
    df['MerchantCategory'] = df['MerchantCategory'].map(cat_map['MerchantCategory'])
    df['TransactionType'] = df['TransactionType'].map(cat_map['TransactionType'])
    df['TransactionFrequency'] = df['TransactionFrequency'].map(cat_map['TransactionFrequency'])
    df['BankName'] = df['BankName'].map(cat_map['BankName'])
    df['UnusualLocation'] = df['UnusualLocation'].map(cat_map['UnusualLocation'])
    df['UnusualAmount'] = df['UnusualAmount'].map(cat_map['UnusualAmount'])
    df['NewDevice'] = df['NewDevice'].map(cat_map['NewDevice'])

    # Convert numeric columns
    numeric_cols = ['Amount', 'Latitude', 'Longitude', 'AvgTransactionAmount', 'FailedAttempts']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values with zero (or use your logic)
    df.fillna(0, inplace=True)

    # Select the columns in the order your model expects
    X = df[['Amount', 'MerchantCategory', 'TransactionType', 'Latitude', 'Longitude',
            'AvgTransactionAmount', 'TransactionFrequency', 'UnusualLocation',
            'UnusualAmount', 'NewDevice', 'FailedAttempts', 'BankName']]

    # Scale features
    X_scaled = scaler.transform(X)

    return X_scaled
