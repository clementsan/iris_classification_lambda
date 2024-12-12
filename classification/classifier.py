from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import os


class Classifier:
    def __init__(self):
        pass

    def train_and_save(self):
        print("\nIRIS model training...")
        iris = load_iris()
        ada = AdaBoostClassifier(n_estimators=5)

        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=42)
        model = ada.fit(X_train, y_train)

        print(f"Model score: {ada.score(X_train, y_train):.3f}")
        print(f"Test Accuracy: {ada.score(X_test, y_test):.3f}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        test_data_csv_path = os.path.join(parent_dir, "data", "test_data.csv")

        pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test, columns=['4'])], axis=1).to_csv(test_data_csv_path,
                                                                                              index=False)

        model_path = os.path.join(parent_dir, "models", "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")


    def load_and_test(self, data):
        print("\nIRIS model prediction...")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, "models", "model.pkl")
        model = joblib.load(model_path)

        features = data.get('features', [])

        if not isinstance(features, list) or not all(isinstance(sample, list) for sample in features):
            raise ValueError("Input data should be a list of feature vectors.")

        # Predict the class
        predictions = model.predict(features).tolist()
        probabilities = model.predict_proba(features).tolist()

        # Map predictions to class labels
        iris_types = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        prediction_labels = [iris_types[pred] for pred in predictions]

        return {"predictions": prediction_labels, "probabilities": probabilities}
