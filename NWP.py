import os
import streamlit as st
import string
import transformers
from transformers import BertTokenizer, BertForMaskedLM
import joblib
import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow import compat
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Optimizer
from docx import Document

def read_word_document(file_path):
    doc = Document(file_path)
    document_text = ""
    for paragraph in doc.paragraphs:
        document_text += paragraph.text + "\n"
    return document_text

#word_document_path = "C:\Users\Sai Tejaswi\OneDrive\Desktop\Minor_project\corpus.docx"

st.set_page_config(page_title='Next Word Prediction Model', page_icon=None, layout='centered', initial_sidebar_state='auto')
#train = read_word_document(word_document_path)

#@st.cache()
def load_model(model_name):
    try:
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      if model_name.lower() == "bert":
            bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
            return bert_tokenizer,bert_model
      elif model_name.lower() == "lstm":
            lstm_model = joblib.load("C:\\Users\\Sai Tejaswi\\OneDrive\\Desktop\\Minor_project\\lstm_model.joblib")
            return bert_tokenizer, lstm_model
      else:
            gru_model = joblib.load("C:\\Users\\Sai Tejaswi\\OneDrive\\Desktop\\Minor_project\\gru_model.joblib")
            return bert_tokenizer, gru_model
    except Exception as e:
        pass


#use joblib to fast your function

def decode(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])

def encode(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx

def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
  input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
  with torch.no_grad():
    predict = bert_model(input_ids)[0]
  bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
  return {'bert': bert}

def get_prediction_eos(input_text):
  try:
    if model_name == 'BERT':
      input_text += ' <mask>'
      res = get_all_predictions(input_text, top_clean=int(top_k))
      return res 
  except Exception as error:
    pass

try:
  st.markdown("<h1 style='text-align: center;'>Next Word Prediction</h1>", unsafe_allow_html=True)
  #st.markdown("<h4 style='text-align: center; color: #B2BEB5;'><i>Keywords  : BertTokenizer, BertForMaskedLM, torch</i></h4>", unsafe_allow_html=True)

  st.sidebar.text("Next Word Prediction Model")
  top_k = st.sidebar.slider("Select How many words do you need", 1 , 100, 1) #some times it is possible to have less words
  print(top_k)
  model_name = st.sidebar.selectbox(label='Model to Apply', options=['BERT', 'LSTM', 'GRU'], index=0,  key = "model_name")
  input_text = st.text_area("Enter your text here")
  tokenizer = Tokenizer()
  answer_as_string = ""
  ans = ""
  #tokenizer.fit_on_texts(train)
  #####################BERT################################
  if model_name == 'BERT':
    bert_tokenizer, bert_model  = load_model(model_name) 
    #click outside box of input text to get result
    res = get_prediction_eos(input_text)
    answer = []
    print(res['bert'].split("\n"))
    for i in res['bert'].split("\n"):
          answer.append(i)
    answer_as_string = "    ".join(answer)
    print(answer_as_string)
   ######################LSTM############################## 
  elif model_name == 'LSTM':
    bert_tokenizer, lstm_model  = load_model(model_name)
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    print(token_list)
    sequence = pad_sequences([token_list], maxlen=72, padding='pre')
    print(sequence)
    lstm_predictions = lstm_model.predict(sequence)
    print(lstm_predictions)
    lstm_top_indices = np.argsort(lstm_predictions[0])[-top_k:][::-1]
    print(lstm_top_indices)
    lstm_predicted_words = [word for word, index in tokenizer.word_index.items() if index in lstm_top_indices]
    print(lstm_predicted_words)
    answer_as_string = "    ".join(lstm_predicted_words)
    for x in lstm_predicted_words:
          print(x)
          ans += x + "    "
    print(answer_as_string)
    print(ans)
  #####################GRU#########################################
  else:  
    bert_tokenizer, gru_model  = load_model(model_name)
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    sequence = pad_sequences([token_list], maxlen=72, padding='pre')
    gru_predictions = gru_model.predict(sequence)
    gru_top_indices = np.argsort(gru_predictions[0])[-top_k:][::-1]
    gru_predicted_words = [word for word, index in tokenizer.word_index.items() if index in gru_top_indices]
    answer_as_string = "    ".join(gru_predicted_words)
    for x in gru_predicted_words:
          print(x)
          ans += x + "    "
    print(answer_as_string)
    print(ans)

  st.text_area("Predicted List is Here",answer_as_string,key="predicted_list")
  st.image('https://freepngimg.com/download/keyboard/6-2-keyboard-png-file.png',use_column_width=True)
  #st.markdown("<h6 style='text-align: center; color: #808080;'>Created By <a href='https://github.com/7Vivek'>Vivek</a> - Checkout complete project <a href='https://github.com/7Vivek/Next-Word-Prediction-Streamlit'>here</a></h6>", unsafe_allow_html=True)

except Exception as e:
  print("Error:", e)
  st.error(f"An error occurred: {e}")
  