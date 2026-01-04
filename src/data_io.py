import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_train_test(train_path, test_path, target_col):
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)

    X_train = train.drop(columns=[target_col]).values
    y_train_raw = train[target_col].values

    X_test = test.drop(columns=[target_col]).values
    y_test_raw = test[target_col].values

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    feature_names = train.drop(columns=[target_col]).columns.to_list()
    return X_train, y_train, X_test, y_test, feature_names
