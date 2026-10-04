import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')

FEATURE_ORDER = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                  'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                  'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                  'V28', 'Amount']

def predict_fraud(transaction: dict):
    try:
        df = pd.DataFrame([transaction], columns=FEATURE_ORDER)
        df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
        probability = model.predict_proba(df)[0][1]
        is_fraud = bool(probability >= 0.5)
        return float(probability), is_fraud
    except Exception as e:
        raise ValueError(f'prediction failed : {str(e)}')

