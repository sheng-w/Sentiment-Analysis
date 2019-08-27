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
    data = pd.read_csv(argv[1])[40000:50000]
    distribution = data.groupby(['Score']).size()    
    print (distribution)
 
if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)
