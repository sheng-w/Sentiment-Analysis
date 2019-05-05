#!/usr/bin/env python3.6
from sys import argv
import pandas as pd
import numpy as np
import csv
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

def modelGen(max_features, max_len):
    #Build the model
    model = Sequential()
    model.add(Embedding(max_features+1, 256, input_length=max_len))
    #model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    #model.add(Dense(5, activation='sigmoid'))

    #Compile the model
    #model.compile(loss='binary_crossentropy',
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def main(argv):
 
    #Read csv data into a dataframe 
    #data = pd.read_csv(argv[1]).sample(10000)
    data = pd.read_csv(argv[1])
       
 
    #Set up the training parameters
    max_features = 40000   #cutoff for num of most common words
    max_len = 300           #cutoff for length of sequence
    batch_size = 500       

    #Convert training text to sequence
    t = Tokenizer(num_words=max_features)
    t.fit_on_texts(data['Text'])
    vocab_size = len(t.word_index)
    x = t.texts_to_sequences(data['Text'])
    #Pad the training sequence length to max_len
    x = sequence.pad_sequences(x, maxlen=max_len)
    #Convert labels to categorical one-hot encoding
    y = to_categorical(data['Score']-1, num_classes=5) 

    #define 5-fold cross validation test harness
    kfold = KFold(n_splits=5, shuffle=True, random_state=np.random.seed())
    cvscores = []
    for train, test in kfold.split(x, y):
        #Generate the model
        model = modelGen(max_features, max_len)
        
        #Fit the model
        model.fit(x[train], y[train], epochs=1, batch_size=batch_size,
                  validation_data=(x[test], y[test])) 

        #Prediction 
        output = model.predict(x[test])
        count = 0
        for i in range(len(test)):
            print (output[i],y[test][i])
            if (np.argmax(output[i]) == np.argmax(y[test][i])):
                count += 1
        print (count*1.0/len(test))
        ##Evaluate the model
        #score = model.evaluate(x[test], y[test], batch_size=batch_size)
        #print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        #cvscores.append(score[1] * 100)
    #print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)
