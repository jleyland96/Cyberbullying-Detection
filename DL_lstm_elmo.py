import tensorflow as tf
import pandas as pd
import csv
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.layers import LSTM, Lambda, Input, Dense
from keras.models import Model, load_model
from keras.engine import Layer
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

validation_results = []
f1_results = []
batch_size = 32
max_len = 32

# Custom callback function
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        print()
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("METRICS")
        print("F1       :", _val_f1)
        print("PRECISION:", _val_precision)
        print("RECALL   :", _val_recall)
        validation_results.append(round(_val_acc, 4))
        f1_results.append(round(_val_f1, 4))

        # Print validation accuracy and f1 scores (so we can plot later)
        print("\nVAL_ACC:\n", validation_results)
        print("\n\n")
        print("F1:\n", f1_results)

        # Save the model for another time
        # save_model(self.model, save_path)
        return


# Create an instance of this callback function
metrics = Metrics()


# Get the original data from my CSV file
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


# Takes a sequence of strings and returns sequence of 1024-dim vectors of ELMo embedding
def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]


def pad_inputs(X):
    new_X = []
    for seq in X:
        new_seq = []
        seq = seq.split()
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_X.append(new_seq)
    return new_X


if __name__ == "__main__":
    sess = tf.Session()
    K.set_session(sess)
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    X, y = get_data(filename='cleaned_tweets_16k.csv')
    X = pad_inputs(X)
    print(X[:3])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    input_text = Input(shape=(max_len,), dtype=tf.string)  #was None,max_len,1024
    embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
    x = LSTM(units=256, recurrent_dropout=0.2, dropout=0.2)(embedding)
    out = Dense(units=1, activation='sigmoid')(x)

    model = Model(input_text, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test),
                        batch_size=batch_size, epochs=5, verbose=1)


