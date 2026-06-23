import streamlit as st
import pickle as pkl
import  string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps=PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
         if i.isalnum(): # if the word is alpha numeric meaning if its a special character
             y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english")  and i not  in string.punctuation: # remove the puntuation marks and stopwords
          y.append(i)
        
    text=y[:]
    y.clear()

    for i in text:
      y.append(ps.stem(i))
    
    return " ".join(y)  


# preprocess
# vectorize
# predict
# display


tfidf= pkl.load(open("tokenizer.pkl","rb"))
model=pkl.load(open("model.pkl","rb"))

st.title("Spam SMS/Email Classifier")
input_msg=st.text_input("Enter Message")

if st.button("Predict"):

    transformed_text=transform_text(input_msg)
    vector_input=tfidf.transform([transformed_text]).toarray()

    result=model.predict(vector_input)[0]


    if result==1:
        st.header('Spam')
    else:
        st.header("Not Spam")
