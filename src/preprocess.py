import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split():
    df = pd.read_csv('data/processed/cleaned.csv')
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    return train_test_split(X, y, test_size=0.2, random_state=42)