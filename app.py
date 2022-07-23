import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
from sklearn.naive_bayes import MultinomialNB


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open('model.pkl','rb'))


st.title ('EMAIL SPAM CLASSIFIER')

input_sms = st.text_input("ENTER THE MESSAGE")

if st.button('Predict'):
    #1 Preprocess
    transform_sms = transform_text(input_sms)

    #2 vectorize
    vector_input = tfidf.transform([transform_sms])

    #3 prdict
    result = model.predict(vector_input)[0]

    #4 result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

