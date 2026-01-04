from sklearn.metrics import accuracy_score, f1_score
from .models import make_model

def evaluate_on_test(Xtr, ytr, Xte, yte, clf):
    model = make_model(clf)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "acc": accuracy_score(yte, pred),
        "f1_macro": f1_score(yte, pred, average="macro"),
    }
