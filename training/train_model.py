import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os
import kagglehub

# 🔽 Download the dataset from Kaggle Hub
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
dataset_path = os.path.join(path, "spam.csv")

# 🔽 Load dataset
df = pd.read_csv(dataset_path, encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 🔽 Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 🔽 Vectorize text
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# 🔽 Train model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# 🔽 Evaluate
y_pred = model.predict(X_test_vect)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ✅ Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n✅ Model and vectorizer saved!")
