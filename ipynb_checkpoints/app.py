from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import re
import string
import sys
import logging
# from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("reviewHotelJakarta.csv", encoding="latin-1")
	df.drop(['Hotel_name', 'name'], axis=1, inplace=True)
	
	# START Prosessing
	# 1. Clean the text
	def clean_text(text):
		return re.sub('[^a-zA-Z]', ' ', text).lower()

	# Labeling based on rating
	df['cleaned_text'] = df['Review'].apply(lambda x: clean_text(x))
	df['label'] = df['Rating'].map({1.0:0, 2.0:0, 3.0:0, 4.0:1, 5.0:1})
	

	# 2. Adding aditional features -> Lenght of Review text, and percentage of punctuations in the Review text
	def count_punct(review):
		count = sum([1 for char in review if char in string.punctuation])
		return round(count/(len(review) - review.count(" ")), 3)*100

	df['Review_len'] = df['Review'].apply(lambda x: len(str(x)) - str(x).count(" "))
	df['punct'] = df['Review'].apply(lambda x: count_punct(str(x)))

	# 3. Tokenization
	def tokenize_text(text):
		tokenized_text = text.split()
		return tokenized_text

	df['tokens'] = df['cleaned_text'].apply(lambda x: tokenize_text(x))

	# 4. Lemmatization and Removing Stopwords
	all_stopwords = stopwords.words('english')
	all_stopwords.remove('not')
	def lemmatize_text(token_list):
		return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])

	lemmatizer = nltk.stem.WordNetLemmatizer()
	df['lemmatized_review'] = df['tokens'].apply(lambda x: lemmatize_text(x))

	# END Processing

	# Feature Extraction (TF-IDF)
	# Extract Feature With CountVectorizer -> Cleaning : convert all of data to lower case and removing all punctuation marks. 
	X = df[['lemmatized_review', 'Review_len', 'punct']]
	y = df['label']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	# ignore terms that occur in more than 50% documents and the ones that occur in less than 2
	tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
	tfidf_train = tfidf.fit_transform(X_train['lemmatized_review'])
	tfidf_test = tfidf.transform(X_test['lemmatized_review'])

	# PREDICTION
	# Using Algorithms Extra Trees Classifier
	classifier = SVC(kernel= 'linear', random_state = 10)
	classifier.fit(tfidf_train, y_train)
	classifier.score(tfidf_test, y_test)





	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = tfidf.transform(data).toarray()
		prediction = classifier.predict(vect)
		return f"{prediction[0]}"

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
	app.run(debug=True)