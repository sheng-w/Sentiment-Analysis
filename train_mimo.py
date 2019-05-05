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

    #Convert helpfulness data
    x_help = data[["HelpfulnessNumerator","HelpfulnessDenominator"]]
    ratio = x_help.apply(lambda x: x.iloc[0]/x.iloc[1] 
                         if x.iloc[1] != 0 else 0, axis=1)
    x_help = pd.concat([x_help, ratio], axis=1).to_numpy()

    #Encode the ordinal label for example
    y = data['Score'].apply(lambda x: [1]*x + [0]*(5-x))
    y = np.asarray(list(y))
    y_one_hot = to_categorical(data['Score']-1, num_classes=5) 
    #print (type(y), y.shape)
    
    text_input = Input(shape=(t_max_len,), name='text_input')
    text_embedding = Embedding(output_dim=200, input_dim=t_max_features, 
                     input_length=t_max_len)(text_input)
    text_lstm = lstm_out = LSTM(64, dropout = 0.2, recurrent_dropout = 0.2)(text_embedding)
    text_output = Dense(5, activation='sigmoid', name='text_output')(text_lstm)
    
    summary_input = Input(shape=(s_max_len,), name='summary_input')
    summary_embedding = Embedding(output_dim=100, input_dim=s_max_features, 
                     input_length=t_max_len)(summary_input)
    summary_lstm = lstm_out = LSTM(32, dropout = 0.2, recurrent_dropout = 0.2)(summary_embedding)
    summary_output = Dense(5, activation='sigmoid', name='summary_output')(summary_lstm)

    help_input = Input(shape=(3,), name='help_input')

    main_input = concatenate([text_lstm, summary_lstm, help_input])
    main_output = Dense(5, activation='softmax', name='main_output')(main_input) 
 
    model = Model(inputs=[text_input, summary_input, help_input], outputs=[main_output, text_output, summary_output]) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['categorical_accuracy']) 
    model.fit([x_text[:split],x_summary[:split], x_help[:split]], [y_one_hot[:split], y[:split], y[:split]], epochs=1, batch_size=batch_size,
              validation_data=([x_text[split:],x_summary[split:], x_help[split:]], [y_one_hot[split:], y[split:], y[split:]]))
     
    output = model.predict([x_text[split:],x_summary[split:], x_help[split:]])
    for i in range(20):
        print (output[0][i], output[1][i], output[2][i], y_one_hot[i])
    #count = 0
    #for i in range(1000):
    #    pred = output[0][i].round()
    #    true = y[split:][i]
    #    #print (pred)
    #    #print (true)
    #    a = 0
    #    for j in range(5):
    #        if pred[j] == true[j]:
    #            a += 1
    #    if a == 5:
    #        count += 1
    #print (count)
 
if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)
