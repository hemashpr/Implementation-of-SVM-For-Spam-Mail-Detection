# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the spam mail dataset using Pandas.
2. Preprocess the mail messages and convert them into TF-IDF feature vectors.
3. Split the dataset into training and testing sets.
4. Train a linear SVM classifier using the training data.
5. Predict the test messages and calculate the classification accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Hemash K
RegisterNumber:  212225050015
*/


# Program to implement SVM for Spam Mail Detection
# Developed by:
# RegisterNumber:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep the required columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# Convert labels to numerical values
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Separate input and output
X = data["message"]
y = data["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train SVM classifier
svm = SVC(kernel="linear")
svm.fit(X_train_tfidf, y_train)

# Predict the test data
y_pred = svm.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("SVM Spam Mail Detection")
print("-----------------------")
print("Accuracy:", accuracy)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

# Test with a new message
message = ["Congratulations! You have won a free lottery prize."]

message_tfidf = vectorizer.transform(message)
prediction = svm.predict(message_tfidf)

if prediction[0] == 1:
    print("\nNew Message: SPAM")
else:
    print("\nNew Message: HAM")
```

## Output:

<img width="731" height="401" alt="image" src="https://github.com/user-attachments/assets/3a19824f-12b1-4d9d-97dc-7a892e3fc7cb" />




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
