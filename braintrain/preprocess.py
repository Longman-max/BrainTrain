import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    def __init__(self, data):
        if isinstance(data, str):
            # If a file path is given, load CSV
            self.df = pd.read_csv(data)
        else:
            # Otherwise assume it's already a DataFrame
            self.df = data.copy()

        self.label_column = "label"
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def preprocess(self):
        # Forward fill missing values
        self.df.fillna(method="ffill", inplace=True)

        # Encode categorical columns
        for col in self.df.select_dtypes(include=["object"]).columns:
            if col != self.label_column:
                self.df[col] = self.encoder.fit_transform(self.df[col])

        # Scale numeric features
        features = self.df.drop(columns=[self.label_column])
        self.df[features.columns] = self.scaler.fit_transform(features)

    def split(self, target=None, test_size=0.2, random_state=42):
        target = target or self.label_column
        X = self.df.drop(columns=[target])
        y = self.df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
