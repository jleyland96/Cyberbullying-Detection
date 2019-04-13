import tensorflow as tf
import pandas as pd
import csv
import tensorflow_hub as hub
from keras import backend as K
from keras.utils import np_utils
import keras.layers as layers
from keras.layers import LSTM, Lambda, Input, Dense, Bidirectional
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
best_confusion_matrix = ""

# Custom callback function
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        global best_confusion_matrix

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
        print("")
        print("F1:\n", f1_results)

        # if the current f1 value is bigger than all previous f1 scores, save the model and matrix
        if (len(f1_results) > 1 and _val_f1 > max(f1_results[:-1])) or (len(f1_results) == 1):
            best_confusion_matrix = confusion_matrix(val_targ, val_predict)

            # Save the model for another time
            print("SAVING NEW MODEL")
            save_model(self.model, SAVE_PATH)
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
        # print("F1 WEIGHTED       :", _val_f1_weighted)
        # print("PRECISION WEIGHTED:", _val_precision_weighted)
        # print("RECALL WEIGHTED   :", _val_recall_weighted)
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
        # print("F1 weighted:\n", f1_results_weighted)
        print("F1 micro:\n", f1_results_micro)

        # Save the model for another time
        # save_model(self.model, save_path)
        if (len(f1_results_micro) > 1 and _val_f1_micro > max(f1_results_micro[:-1])) or (len(f1_results_micro) == 1):
            best_confusion_matrix = confusion_matrix(val_targ, val_predict)

            print("SAVING NEW MODEL")
            # Save the model for another time
            save_model(self.model, SAVE_PATH)
        return


# create an instance of the 3 class metric
three_class_metrics = Three_Class_Metrics()


def save_model(model, path):
    # serialize weights to HDF5
    model.save_weights(str(path) + str(".h5"))
    print("Saved model to disk")


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


# Get the right MAX_LEN for this data
def get_pad_length(filename):
    if filename == "1k":
        return 30
    elif filename == "16k_2class" or filename == "16k_3class":
        return 32  # was 32
    elif filename == "dixon":
        return 50
    else:
        return 32


# F1 loss and calculation
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


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
def print_results(history, y_pred, y_test):
    print("TRAIN:", list(np.round(history.history['acc'], 4)))
    print("TEST:", list(np.round(history.history['val_acc'], 4)))
    print("LOSS:", list(np.round(history.history['loss'], 4)))
    print("Max F1 was", max(f1_results), "at epoch", f1_results.index(max(f1_results)) + 1)
    print("F1:", f1_results)

    # CONFUSION MATRIX
    # print("confusion matrix:\n", confusion_matrix(y_test, y_pred))


# 3 class version of the above function
def print_3class_results(history, y_pred, y_test):
    # PRINT FINAL TRAIN/TEST/LOSS INFO
    print("TRAIN:", list(np.round(history.history['acc'], 4)))
    print("TEST:", list(np.round(history.history['val_acc'], 4)))
    print("LOSS:", list(np.round(history.history['loss'], 4)))
    print("")

    # MAXIMUMS
    # print("F1 WEIGHTED:", list(np.round(f1_results_weighted, 4)))
    # print("Max F1 weighted was", max(f1_results_weighted), "at epoch", f1_results_weighted.index(max(f1_results_weighted)) + 1)
    print("F1 MICRO:", list(np.round(f1_results_micro, 4)))
    print("Max F1 micro was", max(f1_results_micro), "at epoch", f1_results_micro.index(max(f1_results_micro)) + 1, "\n")

    # CONFUSION MATRIX
    # print("confusion matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    dataset = "16k_3class"  # dixon, 1k, 16k_2class, 16k_3class
    SAVE_PATH = "twitter_3class_ELMO"
    LOAD_MODEL = False
    TRAINING = True
    max_len = get_pad_length(dataset)
    print(dataset)
    print(max_len)

    # Create session and get the ELMo embeddings
    sess = tf.Session()
    K.set_session(sess)
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    # Get the data according to the number of classes
    if dataset == "16k_2class":
        X, y = get_data(filename='cleaned_tweets_16k.csv')
        train_max = 12544  # 392 batches of size 32
        test_max = 3136    # 98 batches of size 32
    elif dataset == "16k_3class":
        X, y = get_data(filename='cleaned_tweets_16k_3class.csv')
        train_max = 12544  # 392 batches of size 32
        test_max = 3136    # 98 batches of size 32
    elif dataset == "1k":
        X, y = get_data(filename='cleaned_twitter_1K.csv')
        train_max = 768  # 24 batches of 32
        test_max = 192   # 6 batches of 32
    else:  # data set == "dixon"
        X, y = get_data(filename='cleaned_dixon.csv')
        train_max = 12544  # 1736 batches of 32
        test_max = 3136   # 434 batches of 32

    # pre-preparation of data
    X = pad_inputs(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Take a slice of the data so that we don't have half a batch at the end of the epoch
    X_train = X_train[:train_max]   # 392 batches of size 32
    X_test = X_test[:test_max]      # 98 validation batches of size 32
    y_train = y_train[:train_max]   # one label for each train X
    y_test = y_test[:test_max]      # one label for each test X

    if dataset in ["dixon", "1k", "16k_2class"]:
        print("2class")

        # CREATE MODEL
        input_text = Input(shape=(max_len,), dtype=tf.string)
        embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
        x = LSTM(units=400, recurrent_dropout=0.5, dropout=0.5)(embedding)
        out = Dense(units=1, activation='sigmoid')(x)
        model = Model(input_text, out)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

        if LOAD_MODEL:
            model.load_weights(SAVE_PATH + str('.h5'))

        # FIT THE MODEL
        history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test),
                            batch_size=batch_size, epochs=10, verbose=1, callbacks=[metrics])

        # PRINT RESULTS
        # loss, accuracy = model.evaluate(x=np.array(X_test), y=y_test, verbose=0)
        y_pred = model.predict(x=np.array(X_test))
        y_pred = np.round(y_pred, 0)
        # print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")
        print_results(history, y_pred, y_test)

    elif dataset == "16k_3class":
        print("3class")

        # CONVERT THE TAGS TO CATEGORICAL DATA
        y_train_cat = np_utils.to_categorical(y_train)
        y_test_cat = np_utils.to_categorical(y_test)

        # CREATE THE MODEL
        input_text = Input(shape=(max_len,), dtype=tf.string)
        embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
        x = LSTM(units=512, recurrent_dropout=0.5, dropout=0.5)(embedding)
        out = Dense(units=3, activation='softmax')(x)
        model = Model(input_text, out)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print(model.summary())

        if LOAD_MODEL:
            # load weights if needed
            model.load_weights(SAVE_PATH + str(".h5"))

        if TRAINING:
            # Fit the 3-class model, evaluate, and print training results
            # FIT THE MODEL - 3_class
            history = model.fit(np.array(X_train), y_train_cat, validation_data=(np.array(X_test), y_test_cat),
                                batch_size=batch_size, epochs=10, verbose=1, callbacks=[three_class_metrics])

            # PRINT RESULTS
            loss, accuracy = model.evaluate(x=np.array(X_test), y=y_test_cat, verbose=0)
            y_prob = model.predict(x=np.array(X_test))
            labels_pred = y_prob.argmax(axis=-1)
            print("\bTEST_ACC = " + str(round(accuracy, 4)))
            print_3class_results(history, labels_pred, y_test)
            print(best_confusion_matrix)
            print("Micro Precision = ", round(precision_score(y_test, labels_pred, average='micro'), 4))
            print("Micro Recall = ", round(recall_score(y_test, labels_pred, average='micro'), 4))
            print("Micro F1 = ", round(f1_score(y_test, labels_pred, average='micro'), 4), "\n")
        else:
            # PRINT RESULTS
            loss, accuracy = model.evaluate(x=np.array(X_test), y=y_test_cat, verbose=0)
            y_prob = model.predict(x=np.array(X_test))
            labels_pred = y_prob.argmax(axis=-1)
            print("\bTEST_ACC = " + str(round(accuracy, 4)))
            print("Micro Precision = ", round(precision_score(y_test, labels_pred, average='micro'), 4))
            print("Micro Recall = ", round(recall_score(y_test, labels_pred, average='micro'), 4))
            print("Micro F1 = ", round(f1_score(y_test, labels_pred, average='micro'), 4), "\n")




