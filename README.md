# BrainTrain

Lightweight data processor and training tool.

## Features
- Clean & preprocess datasets  
- Encode categoricals & scale features  
- Train/test split in one step  
- Simple training & evaluation with scikit-learn  

## Install
```bash
git clone https://github.com/Longman-max/BrainTrain.git
cd BrainTrain
pip install -r requirements.txt
````

## Usage

```python
from braintrain.preprocess import DataProcessor
from braintrain.trainer import Trainer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("data.csv")
processor = DataProcessor(df).clean().encode_categoricals().scale()
X_train, X_test, y_train, y_test = processor.split("label")

trainer = Trainer(RandomForestClassifier())
trainer.train(X_train, y_train)
print("Accuracy:", trainer.evaluate(X_test, y_test))
```
## 💬 Join the Discussion
Have ideas, feedback, or questions about BrainTrain?  
Check out our first [GitHub Discussion](../../discussions) and join the conversation!


## License

MIT
