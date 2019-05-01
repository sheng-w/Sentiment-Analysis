#!/usr/bin/env python3.6
from sys import argv
import pandas as pd
import csv

def main(argv):
    data = pd.read_csv(argv[1])

if __name__ == "__main__":
    #np.random.seed(0)
    main(argv)
