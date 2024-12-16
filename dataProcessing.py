import pandas as pd


def load_and_preprocess_data(filepath):
    # Data loading and preprocessing as described in the report
    data = pd.read_csv(filepath)
    
    # Remove non-polypharmacy records (less than 5 medications)
    data['poly'] = data.iloc[:, 1:21].sum(axis=1)
    data = data[data['poly'] >= 5]
    
    # Remove redundant records
    data = data.drop_duplicates(subset=['patient_id', 'hospit'] + list(data.columns[1:21]))
    
    # Prepare features and target
    X = data.iloc[:, 1:21].values
    y = data['hospit'].values
    
    return X, y