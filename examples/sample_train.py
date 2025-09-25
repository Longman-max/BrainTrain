import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from braintrain.preprocess import DataProcessor
from braintrain.trainer import Trainer

df = pd.read_csv("data/sample.csv")

processor = DataProcessor(df, target="label")
df_clean = processor.clean().encode_categoricals().scale().encode_target()
X_train, X_test, y_train, y_test = processor.split()

clf = RandomForestClassifier()
trainer = Trainer(clf)
trainer.train(X_train, y_train)
acc = trainer.evaluate(X_test, y_test)

print(f"Accuracy: {acc:.2f}")

os.makedirs("models", exist_ok=True)
trainer.save("models/braintrain_model.pkl")
