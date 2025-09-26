from braintrain.preprocess import DataProcessor
from braintrain.trainer import ModelTrainer

# Load & preprocess data
processor = DataProcessor("data/sample_data.csv")
processor.preprocess()

X_train, X_test, y_train, y_test = processor.split()

# Train and evaluate
trainer = ModelTrainer()
trainer.train(X_train, y_train)
metrics = trainer.evaluate(X_test, y_test)

print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
