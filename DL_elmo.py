import tensorflow as tf
import pandas as pd
import csv
import tensorflow_hub as hub
import os
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
import numpy as np


def get_data(filename):
    X = []
    Y = []
    print("Getting data from " + filename)

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:

            if line_count == 0:
                pass
            else:
                label_bullying = int(row[0])
                text = row[1]

                X.append(text)
                Y.append(label_bullying)

            line_count += 1

    print("processed", line_count-1, "comments\n")
    return X, y


X, y = get_data("cleaned_tweets_16k.csv")
print(X[:50])
print(y[:50])


