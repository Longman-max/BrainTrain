from sklearn.metrics import accuracy_score
import joblib

class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        """Fit the model on training data."""
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate model performance (accuracy)."""
        preds = self.model.predict(X_test)
        return accuracy_score(y_test, preds)

    def save(self, path="models/braintrain_model.pkl"):
        """Save trained model to disk."""
        joblib.dump(self.model, path)
