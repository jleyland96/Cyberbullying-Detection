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
    y = []
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
                y.append(label_bullying)

            line_count += 1

    print("processed", line_count-1, "comments\n")
    return X, y


# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)
class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# Function to build model
def build_model(): 
  input_text = layers.Input(shape=(1,), dtype="string")
  embedding = ElmoEmbeddingLayer()(input_text)
  dense = layers.Dense(256, activation='relu')(embedding)
  pred = layers.Dense(1, activation='sigmoid')(dense)

  model = Model(inputs=[input_text], outputs=pred)

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  
  return model


# GET ALL THE DATA
X, y = get_data("cleaned_tweets_16k.csv")
data = {}
data['sentence'] = X
data['label'] = y
df = pd.DataFrame.from_dict(data)
print(df.head())

# SPLIT INTO TRAIN/TEST
msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]
print(len(train_df))
print(len(test_df))

# PUT INTO VALID INPUT FORM
train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:30]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['label'].tolist()

test_text = test_df['sentence'].tolist()
test_text = [' '.join(t.split()[0:150]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['label'].tolist()

model = build_model()
model.fit(train_text, 
          train_label,
          validation_data=(test_text, test_label),
          epochs=1,
          batch_size=32)
