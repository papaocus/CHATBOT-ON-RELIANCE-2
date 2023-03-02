import PyPDF2
import os
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


##Then I opened the required File
##Then I read all the file and extracted all the data
pdf_path = os.path.abspath("D:\Reliance chatbot 3\AR_2021-22.pdf")

with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ''
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text += page.extract_text()
##Then I initialize an pre trained instance using BERT 


model = SentenceTransformer('bert-base-nli-mean-tokens')

#Then I initilize a function that uses regular expression to preprocess the text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

##This function takes takes the text and tokenizes each sentence 
def tokenize_text(text):
    sentences = text.split('.')
    tokenized_sentences = [sentence.split(' ') for sentence in sentences]
    return tokenized_sentences

## this function takes three arguments input_text, tokenized text model the function uses
## a pre-trained model to compute cosine similarity and returns the most similar sentence

def generate_response(input_text, tokenized_text, model):
    input_embedding = model.encode(input_text).reshape(1, -1)
    similarity_scores = []
    for sentence_tokens in tokenized_text:
        sentence_embedding = model.encode(' '.join(sentence_tokens)).reshape(1, -1)
        score = cosine_similarity(input_embedding, sentence_embedding)[0][0]
        similarity_scores.append(score)
    most_similar_sentence_index = similarity_scores.index(max(similarity_scores))
    response = tokenized_text[most_similar_sentence_index + 1]
    return ' '.join(response)

def handle_input(input_text, tokenized_text, model):
    response_text = generate_response(input_text, tokenized_text, model)
    return response_text

## Then in the end I have initialized a Streamlit app
def app():
    st.set_page_config(page_title='Reliance Annual Report ChatBot')
    st.title('Reliance Annual Report ChatBot')
    st.markdown('Ask me anything about Reliance!')

    preprocessed_text = preprocess_text(text)
    tokenized_text = tokenize_text(preprocessed_text)

    while True:
        input_text = st.text_input('You', '')
        if input_text:
            response_text = handle_input(input_text, tokenized_text, model)
            st.text_area('Reliance ChatBot', value=response_text, height=200)

if __name__ == '__main__':
    app()


        


