#!/usr/bin/env python3.6
from sys import argv
import pandas as pd
import csv
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.utils import to_categorical


def main(argv):
 
    #Read csv data into a dataframe 
    data = pd.read_csv(argv[1])
    #Convert text to list of integers
    t = Tokenizer()
    t.fit_on_texts(data['Text'])
    vocab_size = len(t.word_index)
    print (vocab_size)
    x = t.texts_to_sequences(data['Text'])
    ##xx = data['Text'].apply(lambda text: t.texts_to_sequences(text_to_word_sequence(text)))
    y = to_categorical(data['Score']-1, num_classes=5)       


 
    #Split data into 80% training data and 20% testing data
    x_train = x[:4000]
    x_test = x[4000:]
    y_train = y[:4000]
    y_test = y[4000:] 

    #Set up the training parameters
    ##max_features = 20000   #cutoff for num of most common words
    max_len = 80           #cutoff for length of sequence
    batch_size = 32        

    #Pad squence to the same length
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    #Build the model
    model = Sequential()
    model.add(Embedding(vocab_size+1, 300, input_length=max_len))
    model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
   
    #Train
    model.fit(x_train, y_train, epochs=15, batch_size=batch_size,
              validation_data=(x_test, y_test)) 
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print (score)
    print (acc)

if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)
