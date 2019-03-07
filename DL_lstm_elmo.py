import tensorflow as tf
import pandas as pd
import csv
import tensorflow_hub as hub
from keras import backend as K
from keras.utils import np_utils
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
f1_results_micro = []
f1_results_weighted = []
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


# Custom callback function for 3 classes
class Three_Class_Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s_weighted = []
        self.val_recalls_weighted = []
        self.val_precisions_weighted = []
        self.val_f1s_micro = []
        self.val_recalls_micro = []
        self.val_precisions_micro = []

    def on_epoch_end(self, epoch, logs={}):
        print()
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        # VALIDATION ACCURACY
        _val_acc = accuracy_score(val_targ, val_predict)
        validation_results.append(round(_val_acc, 4))

        # GET WEIGHTED METRICS
        _val_f1_weighted = f1_score(val_targ, val_predict, average='weighted')
        _val_recall_weighted = recall_score(val_targ, val_predict, average='weighted')
        _val_precision_weighted = precision_score(val_targ, val_predict, average='weighted')
        self.val_f1s_weighted.append(_val_f1_weighted)
        self.val_recalls_weighted.append(_val_recall_weighted)
        self.val_precisions_weighted.append(_val_precision_weighted)
        print("F1 WEIGHTED       :", _val_f1_weighted)
        print("PRECISION WEIGHTED:", _val_precision_weighted)
        print("RECALL WEIGHTED   :", _val_recall_weighted)
        f1_results_weighted.append(round(_val_f1_weighted, 4))
        print("\n")

        # GET MICRO METRICS
        _val_f1_micro = f1_score(val_targ, val_predict, average='micro')
        _val_recall_micro = recall_score(val_targ, val_predict, average='micro')
        _val_precision_micro = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s_micro.append(_val_f1_micro)
        self.val_recalls_micro.append(_val_recall_micro)
        self.val_precisions_micro.append(_val_precision_micro)
        print("F1 MICRO       :", _val_f1_micro)
        print("PRECISION MICRO:", _val_precision_micro)
        print("RECALL MICRO  :", _val_recall_micro)
        f1_results_micro.append(round(_val_f1_micro, 4))

        # Print validation accuracy and f1 scores (so we can plot later)
        print("\nVAL_ACC:\n", validation_results)
        print("\n\n")
        print("F1 weighted:\n", f1_results_weighted)
        print("F1 micro:\n", f1_results_micro)

        # Save the model for another time
        # save_model(self.model, save_path)
        return


# create an instance of the 3 class metric
three_class_metrics = Three_Class_Metrics()


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


# pad the inputs so that they are of the same length
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


# print the results so that we can plot a graph of the f1/acc/val_acc over time
def print_results(history):
    print("TRAIN:", list(np.round(history.history['acc'], 4)), "\n")
    print("TEST:", list(np.round(history.history['val_acc'], 4)), "\n")
    print("LOSS:", list(np.round(history.history['loss'], 4)), "\n")
    print("Max F1 was", max(f1_results), "at epoch", f1_results.index(max(f1_results)) + 1, "\n")
    print("F1:", f1_results)


# 3 class version of the above function
def print_3class_results(history):
    # PRINT FINAL TRAIN/TEST/LOSS INFO
    print("TRAIN:", list(np.round(history.history['acc'], 4)), "\n")
    print("TEST:", list(np.round(history.history['val_acc'], 4)), "\n")
    print("LOSS:", list(np.round(history.history['loss'], 4)), "\n")
    print("\n")

    # MAXIMUMS
    print("Max F1 weighted was", max(f1_results_weighted), "at epoch", f1_results_weighted.index(max(f1_results_weighted)) + 1, "\n")
    print("Max F1 micro was", max(f1_results_micro), "at epoch", f1_results_micro.index(max(f1_results_micro)) + 1, "\n")


if __name__ == "__main__":
    num_classes = 2

    sess = tf.Session()
    K.set_session(sess)
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    if num_classes == 2:
        X, y = get_data(filename='cleaned_tweets_16k.csv')
    elif num_classes == 3:
        X, y = get_data(filename='cleaned_tweets_16k_3class.csv')

    X = pad_inputs(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Take a slice of the data so that we don't have half a batch at the end of the epoch
    X_train = X_train[:12544]   # 392 batches of size 32
    X_test = X_test[:3136]      # 98 validation batches of size 32
    y_train = y_train[:12544]   # one label for each train X
    y_test = y_test[:3136]      # one label for each test X

    if num_classes == 2:
        # CREATE MODEL
        input_text = Input(shape=(max_len,), dtype=tf.string)
        embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
        x = LSTM(units=256, recurrent_dropout=0.5, dropout=0.5)(embedding)
        out = Dense(units=1, activation='sigmoid')(x)
        model = Model(input_text, out)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # FIT THE MODEL
        history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test),
                            batch_size=batch_size, epochs=2, verbose=1, callbacks=[metrics])

        # PRINT RESULTS
        loss, accuracy = model.evaluate(x=X_test, y=y_test, verbose=0)
        print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")
        print_results(history)

    elif num_classes == 3:
        # CONVERT THE TAGS TO CATEGORICAL DATA
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        # CREATE THE MODEL
        input_text = Input(shape=(max_len,), dtype=tf.string)
        embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
        x = LSTM(units=256, recurrent_dropout=0.5, dropout=0.5)(embedding)
        out = Dense(units=3, activation='softmax')(x)
        model = Model(input_text, out)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # FIT THE MODEL
        history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test),
                            batch_size=batch_size, epochs=2, verbose=1, callbacks=[three_class_metrics])

        # PRINT RESULTS
        loss, accuracy = model.evaluate(x=X_test, y=y_test, verbose=0)
        print("\bTEST_ACC = " + str(round(accuracy * 100, 2)) + "%")
        print_3class_results(history)


