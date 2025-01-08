"""
IRIS Classification - class definition
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Classifier:
    """Classifier class - ML training and testing"""

    def __init__(self):
        pass

    def train_and_save(self):
        """ML training and saving"""
        print("\nIRIS model training...")
        iris = load_iris()
        cart = DecisionTreeClassifier(max_depth=3)

        x_train, x_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.1, random_state=42
        )
        model = cart.fit(x_train, y_train)

        print(f"Model score: {cart.score(x_train, y_train):.3f}")
        print(f"Test Accuracy: {cart.score(x_test, y_test):.3f}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        test_data_csv_path = os.path.join(parent_dir, "data", "test_data.csv")

        pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test, columns=["4"])], axis=1).to_csv(
            test_data_csv_path, index=False
        )

        model_path = os.path.join(parent_dir, "models", "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    def load_and_test(self, data):
        "ML loading and testing"
        print("\nIRIS model prediction...")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, "models", "model.pkl")
        model = joblib.load(model_path)

        features = np.array(data)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        if features.shape[-1] != 4:
            raise ValueError("Expected 4 features per input.")

        # Predict the class
        predictions = model.predict(features).tolist()
        probabilities = model.predict_proba(features).tolist()

        # Map predictions to class labels
        iris_types = {0: "setosa", 1: "versicolor", 2: "virginica"}
        prediction_labels = [iris_types[pred] for pred in predictions]

        return {"predictions": prediction_labels, "probabilities": probabilities}
