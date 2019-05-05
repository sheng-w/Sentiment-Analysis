#!/usr/bin/env python3.6
from sys import argv
import pandas as pd
import numpy as np
import csv
import xgboost as xgb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, concatenate
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def text_model(t_max_len, t_max_features):
    text_input = Input(shape=(t_max_len,), name='text_input')
    text_embedding = Embedding(output_dim=200, input_dim=t_max_features, 
                     input_length=t_max_len)(text_input)
    text_lstm = lstm_out = LSTM(64, dropout = 0.2, recurrent_dropout = 0.2)(text_embedding)
    text_output = Dense(5, activation='sigmoid', name='text_output')(text_lstm)
    model = Model(inputs=[text_input], outputs=[text_output]) 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy']) 
    return model

def summary_model(s_max_len, s_max_features):
    summary_input = Input(shape=(s_max_len,), name='summary_input')
    summary_embedding = Embedding(output_dim=100, input_dim=s_max_features, 
                     input_length=s_max_len)(summary_input)
    summary_lstm = lstm_out = LSTM(32, dropout = 0.2, recurrent_dropout = 0.2)(summary_embedding)
    summary_output = Dense(5, activation='sigmoid', name='summary_output')(summary_lstm)
    model = Model(inputs=[summary_input], outputs=[summary_output]) 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy']) 
    return model

def main_model():
    main_input = Input(shape=(13,), name='main_input')
    main_output = Dense(5, activation='softmax', name='main_output')(main_input)
    model = Model(inputs=[main_input], outputs=[main_output]) 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy']) 
    return model

def main(argv):
 
    #Read csv data into a dataframe 
    #data = pd.read_csv(argv[1]).sample(10000)
    data = pd.read_csv(argv[1])[:20000]
    split = round(data.shape[0]*0.8)
    print (split)    
   
    #Set up the training parameters
    t_max_features = 30000   #cutoff for num of most common words for text
    t_max_len = 300          #cutoff for length of sequence for text
    s_max_features = 30000   #cutoff for num of most common words for summary
    s_max_len = 50          #cutoff for length of sequence for summary
    batch_size = 50       

    #Convert training text to sequence
    t = Tokenizer(num_words=t_max_features)
    t.fit_on_texts(data['Text'])
    x_text = t.texts_to_sequences(data['Text'].astype(str))
    #Pad the training sequence length to max_len
    x_text = sequence.pad_sequences(x_text, maxlen=t_max_len)

    #Convert traing summary to sequency 
    t = Tokenizer(num_words=s_max_features)
    t.fit_on_texts(data['Text'])
    x_summary = t.texts_to_sequences(data['Text'].astype(str))
    #Pad the training sequence length to max_len
    x_summary = sequence.pad_sequences(x_summary, maxlen=s_max_len)

    #Convert
    x_help = data[["HelpfulnessNumerator","HelpfulnessDenominator"]]
    ratio = x_help.apply(lambda x: x.iloc[0]/x.iloc[1] 
                         if x.iloc[1] != 0 else 0, axis=1)
    x_help = pd.concat([x_help, ratio], axis=1).to_numpy()

    #Encode the ordinal label for example
    y = data['Score'].apply(lambda x: [1]*x + [0]*(5-x))
    y = np.asarray(list(y))
    y_one_hot = to_categorical(data['Score']-1, num_classes=5) 

 
    t_model = text_model(t_max_len, t_max_features)
    s_model = summary_model(s_max_len, s_max_features)    
    t_model.fit(x_text[:split],y[:split],epochs=1, batch_size=batch_size) 
    s_model.fit(x_summary[:split],y[:split],epochs=1, batch_size=batch_size) 

    t_output = t_model.predict(x_text)
    s_output = s_model.predict(x_summary)
    
    x_total = np.concatenate([t_output, s_output, x_help], axis=1)   

    m_model = main_model()
    m_model.fit(x_total[:split],y_one_hot[:split],epochs=1, batch_size=batch_size,
                validation_data=(x_total[split:], y_one_hot[split:]))
     
    output = m_model.predict(x_total[split:])
    for i in range(20):
        print (output[i], y_one_hot[i])
     
     
    #param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softmax' }
    #num_round = 10
    #bst = xgb.train(param, data = t_output, label = y_score[:split], num_round)
 
    
    #main_input = concatenate([text_lstm, summary_lstm])
    #main_output = Dense(5, activation='softmax', name='main_output')(main_input) 
 

    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['categorical_accuracy']) 
    #model.fit([x_text[:split],x_summary[:split]], [y[:split], y[:split], y[:split]], epochs=1, batch_size=batch_size,
              #validation_data=([x_text[split:],x_summary[split:]], [y[split:], y[split:], y[split:]]))
     
 
if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)
