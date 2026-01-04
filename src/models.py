from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def build_classifiers(knn_k=5, rf_estimators=200, seed=42):
    return {
        "RF":  RandomForestClassifier(n_estimators=rf_estimators, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=knn_k),
        "GNB": GaussianNB(),
    }

def make_model(clf):
    # KNN ve GNB ölçekten faydalanır; RF genelde gerekmez
    if isinstance(clf, (KNeighborsClassifier, GaussianNB)):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return clf
