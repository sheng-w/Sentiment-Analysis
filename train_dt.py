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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def text_model(t_max_len, t_max_features):
    text_input = Input(shape=(t_max_len,), name='text_input')
    text_embedding = Embedding(output_dim=200, input_dim=t_max_features, 
                     input_length=t_max_len)(text_input)
    text_lstm_sub = LSTM(64, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)(text_embedding)
    text_lstm = LSTM(64, dropout = 0.2, recurrent_dropout = 0.2)(text_lstm_sub)
    text_output = Dense(4, activation='sigmoid', name='text_output')(text_lstm)
    model = Model(inputs=[text_input], outputs=[text_output]) 
    #adam = Adam(lr=0.01)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics = ['accuracy']) 
    return model

def summary_model(s_max_len, s_max_features):
    summary_input = Input(shape=(s_max_len,), name='summary_input')
    summary_embedding = Embedding(output_dim=100, input_dim=s_max_features, 
                     input_length=s_max_len)(summary_input)
    summary_lstm_sub = LSTM(32, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)(summary_embedding)
    summary_lstm = LSTM(32, dropout = 0.2, recurrent_dropout = 0.2)(summary_lstm_sub)
    summary_output = Dense(4, activation='sigmoid', name='summary_output')(summary_lstm)
    model = Model(inputs=[summary_input], outputs=[summary_output]) 
    #loss_weights = {0: 0.2, 1: 0.4, 2: 0.6, 3:0.8, 4:1.0}
    #adam = Adam(lr=0.01)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics = ['accuracy']) 
    return model

def main(argv):
 
    #Read csv data into a dataframe 
    #data = pd.read_csv(argv[1]).sample(10000)
    data = pd.read_csv(argv[1])
    split = round(data.shape[0]*0.8)
    print (split)    
   
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

    #Convert
    x_help = data[["HelpfulnessNumerator","HelpfulnessDenominator"]]
    ratio = x_help.apply(lambda x: x.iloc[0]/x.iloc[1] 
                         if x.iloc[1] != 0 else 0, axis=1)
    x_help = pd.concat([x_help, ratio], axis=1).to_numpy()

    #Encode the ordinal label for example
    y = data['Score'].apply(lambda x: [1]*(x-1) + [0]*(5-x))
    y = np.asarray(list(y))
    y_label = (data['Score']-1).to_numpy()

    #Fit the LSTM model for text and summary individually
    t_model = text_model(t_max_len, t_max_features)
    s_model = summary_model(s_max_len, s_max_features)    
    t_model.fit(x_text[:split],y[:split],epochs=1, batch_size=batch_size) 
    s_model.fit(x_summary[:split],y[:split],epochs=1, batch_size=batch_size) 

    t_output = t_model.predict(x_text)
    s_output = s_model.predict(x_summary)
    ##Predict LSTM output for text and summary
    #t_output = t_model.predict(x_text[16000:16050])
    #s_output = s_model.predict(x_summary[16000:16050])
    for i in range(50):
        print (t_output[split:][i], s_output[split:][i], y[split:][i])  
  
    #Generate input for decision tree
    x_total = np.concatenate([t_output, s_output, x_help], axis=1)   
    d_train = xgb.DMatrix(x_total[:split], label=y_label[:split])
    d_test = xgb.DMatrix(x_total[split:], label=y_label[split:])
   
    #Fit decision tree
    param = {'max_depth': 5, 'eta': 0.5, 'objective': 'multi:softmax', 
             'num_class': 5, 'tree_method': 'gpu_exact'} 
    param['nthread'] = 4
    param['eval_metric'] = 'merror'
    evallist = [(d_test, 'eval'), (d_train, 'train')]
    num_round = 10
    bst = xgb.train(param, d_train, num_round, evallist)
    preds = bst.predict(d_test)
 
if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)
