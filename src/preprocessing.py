import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import json

def fill_missing_values(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(
        include=['object', 'string', 'category']
    ).columns

    num_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        dummy_na=True
    )

    return df

def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0
    )
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    print('Before upsampling count of label 0 {}'.format(sum(y_train == 0)))
    print('Before upsampling count of label 1 {}'.format(sum(y_train == 1)))

    sm = SMOTE(sampling_strategy=1, random_state=1)

    X_train_s, y_train_s = sm.fit_resample(X_train, y_train)

    print('After upsampling count of label 0 {}'.format(sum(y_train_s == 0)))
    print('After upsampling count of label 1 {}'.format(sum(y_train_s == 1)))
    return X_train_s, y_train_s

def save_columns(X):
    columns = {
        'data_columns': [
            col.lower()
            for col in X.columns
        ]
    }
    with open(
        "models/columns.json",
        "w"
    ) as f:
        json.dump(columns, f, indent=4)
    return columns