import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df.copy()
        self.target = target

    def clean(self):
        """Drop duplicates and fill missing values."""
        self.df.drop_duplicates(inplace=True)
        self.df = self.df.ffill()
        return self

    def encode_categoricals(self):
        """Encode categorical features using LabelEncoder, but skip target."""
        for col in self.df.select_dtypes(include=["object"]).columns:
            if col != self.target:
                self.df[col] = LabelEncoder().fit_transform(self.df[col])
        return self

    def encode_target(self):
        """Label-encode target column if it's categorical."""
        if self.target and self.df[self.target].dtype == "object":
            self.df[self.target] = LabelEncoder().fit_transform(self.df[self.target])
        return self

    def scale(self):
        """Standardize numerical features, excluding target."""
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if self.target in num_cols:
            num_cols = num_cols.drop(self.target)
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        return self

    def split(self, test_size=0.2, random_state=42):
        """Split into train/test sets."""
        if not self.target:
            raise ValueError("Target column must be set before splitting.")
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
