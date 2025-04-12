#required libraries
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#loading dataset:
true_df = pd.read_csv("dataset/True.csv")
false_df = pd.read_csv("dataset/Fake.csv")

#labels
true_df['label'] = 1
false_df['label'] = 0
#and combine
df = pd.concat([true_df, false_df], ignore_index=True)

#cleaning the acquired data:
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in stop_words)

    return text
df['content'] = df['text'].apply(clean_text)

#now converting, clean text into numbers
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['content'])

y = df['label']

#splitting dataset:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)


MlModel = LogisticRegression()
MlModel.fit(X_train, y_train) 


y_pred = MlModel.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print('\n\ninput your own model for testing')
input_article = input('Enter article: ')

cleaned_input = clean_text(input_article)
vector_input = vectorizer.transform([cleaned_input])
prediction = MlModel.predict(vector_input)

if prediction[0] == 1:
    print('whoa, that article seems real!')
else: 
    print("haha good one, the article's fake!")
