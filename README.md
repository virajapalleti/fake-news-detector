# fake-news-detector
AI/ML project to detect fake news from correct ones.

dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Working:
- merge the true and fake datasets downloaded from Kaggle
- clean the text (remove symbols, lowercase everything, remove stopwords)
- Use TF-IDF to convert words into numbers
- Split the data: 80% for training, 20% for testing
- model type: logistic regression
