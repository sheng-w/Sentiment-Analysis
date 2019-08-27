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

    #Analyze text
    t = Tokenizer()
    t.fit_on_texts(data['Summary'].astype(str))
    
    #Calculate the vocabulary size 
    vocab_size = len(t.word_index)
    print ("Number of words...")
    print (vocab_size)
 
    #Calculate the frenquency of words based on their occurrence 
    print ("Count the number of words for each frequency...")
    word_counts = pd.DataFrame.from_dict(t.word_counts, 
                  orient='index',columns=['occurrence'])
    counts = word_counts.groupby(['occurrence']).size()
    with open("frequency_summary.txt","w") as output:
        for item in counts.iteritems():
            output.write("%8d%8d\n" %(item[0], item[1]))

    print ("Count the number of texts for each length...")
    sequences = t.texts_to_sequences(data['Text'])
    lengths = [len(x) for x in sequences] 
    length_count = pd.DataFrame({'length':lengths}).groupby('length').size() 
    with open("length_summary.txt","w") as output:
        for item in length_count.iteritems():
            output.write("%8d%8d\n" %(item[0], item[1]))
 
if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)
