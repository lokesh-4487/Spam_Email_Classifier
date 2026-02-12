# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Select Required Columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert Labels to Numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# Convert Text to Numbers
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test Custom Message
msg = ["Congratulations! You won a free lottery ticket"]
msg_vector = vectorizer.transform(msg)
prediction = model.predict(msg_vector)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")
