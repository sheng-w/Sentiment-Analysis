

import numpy as np
import pandas as pd
import csv

import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Embedding, LSTM
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



##----------------------------------- Set Parameters -----------------------------------##
maxlen = 100

max_features = 1024

batch_size = 32


##----------------------------------- Read Data -----------------------------------##
data = pd.read_csv('Review_sub.csv')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Text'])
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

X = tokenizer.texts_to_sequences(data['Text'])
Y = to_categorical(data['Score']-1, num_classes=5)

X = np.array(X)


##----------------------------------- Split Data into Train and Test -----------------------------------##
index_train, index_test = train_test_split(range(len(data)), test_size = 0.2)

X_train, X_test = X[index_train], X[index_test]
Y_train, Y_test = Y[index_train], Y[index_test]

print(len(index_train), 'train sequences')
print(len(index_test), 'test sequences')

# Compute the max lenght of a text
MAX_SEQ_LENGHT = len(max(X_train, key=len)) # 944

##----------------------------------- Pad Sequences -----------------------------------##
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)



##----------------------------------- Define Model -----------------------------------##
model = Sequential()
model.add(Embedding(vocab_size, output_dim=256))
model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))

print(model.summary())


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



#model.fit(X_train, Y_train, batch_size=batch_size, epochs=5, validation_data=(X_test, Y_test))
#score, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size)


#print(score)
#print(accuracy)



##----------------------------------- K-Fold Cross Validation -----------------------------------##

def temp_model():
	model = Sequential()
	model.add(Embedding(vocab_size, output_dim=256))
	model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(5, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
	return model

estimator = KerasRegressor(build_fn=temp_model, epochs=5, batch_size=32, verbose=0)



kfold = KFold(n_splits=2)
results = cross_val_score(estimator, X, Y, cv = kfold, error_score='raise')
# Have some Padding issue
print(results)

#for train_index_cv, test_index_cv in kf.split(X):
	#X_train, X_test = X[train_index_cv], X[test_index_cv]
	#Y_train, Y_test = Y[train_index_cv], Y[test_index_cv]











