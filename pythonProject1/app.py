import nltk
import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stopwords.words('english')


ps = PorterStemmer()


def transform(text):
    text = text.lower()  # lower case
    text = nltk.word_tokenize(text)  # tokenize

    y = []  # removing special characters
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

    for i in text:  # stemming(meaning : loving -> love)
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

