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
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

def modelGen(t_max_len, s_max_len, t_max_features, s_max_features):
    text_input = Input(shape=(t_max_len,), name='text_input')
    text_embedding = Embedding(output_dim=200, input_dim=t_max_features, 
                     input_length=t_max_len)(text_input)
    text_lstm = lstm_out = LSTM(64, dropout = 0.2, recurrent_dropout = 0.2)(text_embedding)
    #text_lstm_sub = LSTM(64, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)(text_embedding)
    #text_lstm = LSTM(64, dropout = 0.2, recurrent_dropout = 0.2)(text_lstm_sub)
    text_output = Dense(1, activation='sigmoid', name='text_output')(text_lstm)
    
    summary_input = Input(shape=(s_max_len,), name='summary_input')
    summary_embedding = Embedding(output_dim=100, input_dim=s_max_features, 
                     input_length=t_max_len)(summary_input)
    summary_lstm = lstm_out = LSTM(32, dropout = 0.2, recurrent_dropout = 0.2)(summary_embedding)
    #summary_lstm_sub = LSTM(32, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)(summary_embedding)
    #summary_lstm = LSTM(32, dropout = 0.2, recurrent_dropout = 0.2)(summary_lstm_sub)
    summary_output = Dense(1, activation='sigmoid', name='summary_output')(summary_lstm)

    help_input = Input(shape=(3,), name='help_input')

    main_input = concatenate([text_lstm, summary_lstm, help_input])
    main_output = Dense(1, activation='sigmoid', name='main_output')(main_input) 
 
    model = Model(inputs=[text_input, summary_input, help_input], outputs=[main_output, text_output, summary_output]) 

    #adam = Adam(lr=0.01)
    model.compile(optimizer="adam", loss='binary_crossentropy', 
                  metrics = ['binary_accuracy']) 
    
    return model

def main(argv):
 
    #Read csv data into a dataframe 
    #data = pd.read_csv(argv[1]).sample(10000)
    data = pd.read_csv(argv[1]).sample(n=100000, replace=False)
   
    #Set up the training parameters
    t_max_features = 30000   #cutoff for num of most common words for text
    t_max_len = 300          #cutoff for length of sequence for text
    s_max_features = 30000   #cutoff for num of most common words for summary
    s_max_len = 50          #cutoff for length of sequence for summary
    batch_size = 500       

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
    y = data['Score'].apply(lambda x: int(x >= 4)).to_numpy()
   
    #define 5-fold cross validation test harness
    kfold = KFold(n_splits=5, shuffle=True, random_state=np.random.seed())
    cvscores = [[] for x in range(6)]
    for train, test in kfold.split(x_text,y):
        model = modelGen(t_max_len, s_max_len, t_max_features, t_max_features) 
        model.fit([x_text[train],x_summary[train], x_help[train]], [y[train], y[train], y[train]], epochs=2, batch_size=batch_size,
              validation_data=([x_text[test],x_summary[test], x_help[test]], [y[test], y[test], y[test]]))

        #Evaluate the model
        score = model.evaluate([x_text[test], x_summary[test], x_help[test]],[y[test], y[test], y[test]], batch_size=batch_size)
        #print("%s: %.2f%% %.2f%% %.2f%%" % ("accuracy", score[-3]*100, score[-2]*100, score[-1]*100))
        cvscores[0].append(score[-6] * 100)
        cvscores[1].append(score[-5] * 100)
        cvscores[2].append(score[-4] * 100)
        cvscores[3].append(score[-3] * 100)
        cvscores[4].append(score[-2] * 100)
        cvscores[5].append(score[-1] * 100)
    print("main train output %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[0]), np.std(cvscores[0])))
    print("text train output %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[1]), np.std(cvscores[1])))
    print("summary train output %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[2]), np.std(cvscores[2])))
    print("main cross validation output %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[3]), np.std(cvscores[3])))
    print("text cross validation output %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[4]), np.std(cvscores[4])))
    print("summary cross validation output %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[5]), np.std(cvscores[5])))
     
 
if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)

