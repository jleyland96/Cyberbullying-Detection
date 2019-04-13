import csv
import os
import numpy as np
import random
from numpy import asarray, zeros
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers import ReLU
from keras.layers import Conv1D
from keras.layers import MaxPool1D, MaxPooling1D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers import GlobalMaxPool1D
from keras.models import model_from_json
from keras.models import Model
from keras.initializers import Constant
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras import optimizers
import keras.backend as K
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf


# 'global' variables
validation_results = []
f1_results = []
f1_results_weighted = []
f1_results_micro = []
max_f1 = 0
max_f1_micro = 0
max_f1_weighted = 0
best_confusion_matrix = ""


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
        print("Best F1 so far", max(f1_results), "\n")

        # if the current f1 value is bigger than all previous f1 scores, save the model and matrix
        if (len(f1_results) > 1 and _val_f1 > max(f1_results[:-1])) or (len(f1_results) == 1):
            best_confusion_matrix = confusion_matrix(val_targ, val_predict)

            # Save the model for another time
            print("SAVING NEW MODEL")
            save_model(self.model, SAVE_PATH)

        print()
        print(confusion_matrix(val_targ, val_predict))
        return


metrics = Metrics()


class Three_Class_Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s_weighted = []
        self.val_recalls_weighted = []
        self.val_precisions_weighted = []
        self.val_f1s_micro = []
        self.val_recalls_micro = []
        self.val_precisions_micro = []

    def on_epoch_end(self, epoch, logs={}):
        global best_confusion_matrix

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

        if (len(f1_results_micro) > 1 and _val_f1_micro > max(f1_results_micro[:-1])) or (len(f1_results_micro)==1):
            print("SAVING NEW MODEL")
            # Save the model for another time
            best_confusion_matrix = confusion_matrix(val_targ, val_predict)
            save_model(self.model, SAVE_PATH)

        return


three_class_metrics = Three_Class_Metrics()


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

                # print(label_bullying)
                # print(text)
                # print("\n")

                X.append(text)
                y.append(label_bullying)

            line_count += 1

    print("processed", line_count-1, "comments\n")
    return X, y


def count_vocab_size(x):
    vec = TfidfVectorizer()
    corpus_fit_transform = vec.fit_transform(x)
    return corpus_fit_transform.shape[1]


def get_glove_matrix_from_dump():
    embedding_matrix = pickle.load(open('embedding_matrices/' + str(matrix) + '.p', 'rb'))
    return embedding_matrix


def get_glove_matrix(vocab_size, t):
    glove_vector_size = 300

    # load embeddings into memory
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("LOADED", len(embeddings_index), "WORD VECTORS")

    # create weight matrix for words in documents
    embedding_matrix = zeros((vocab_size, glove_vector_size))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    if not(file == "cleaned_dixon.csv"):
        pickle.dump(embedding_matrix, open('embedding_matrices/' + str(matrix) + '.p', 'wb'), protocol=2)

    return embedding_matrix


def get_pad_length(filename):
    if filename == "cleaned_text_messages.csv":
        return 32
    elif filename == "cleaned_twitter_1K.csv":
        return 30
    elif filename == "cleaned_formspring.csv":
        return 100
    elif filename == "cleaned_tweets_16k.csv" or filename == "cleaned_tweets_16k_3class.csv":
        return 32  # was 32
    elif filename == "cleaned_dixon.csv":
        return 500
    else:
        return 32


def save_model(model, path):
    if not(file == "cleaned_dixon.csv"):
        # serialize model to JSON
        model_json = model.to_json()
        with open("saved_models/" + str(path) + ".json", "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("saved_models/" + str(path) + ".h5")
    print("Saved model to disk")


def load_model(path):
    # load json and create model
    json_file = open("saved_models/" + str(path) + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/" + str(path) + ".h5")
    print("Loaded model from disk")
    return loaded_model


def draw_graph(history, classes=2):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    # Plot the correct F1 score
    if classes == 2:
        plt.plot(f1_results)
    else:
        plt.plot(f1_results_micro)

    plt.plot(history.history['loss'])
    plt.title('model accuracy and f1 score')
    plt.ylabel('accuracy/f1/loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'f1', 'loss'], loc='lower right')
    plt.savefig('DL_graphs/' + str(SAVE_PATH) + ' graph.png')
    # plt.show()


def print_results(history, y_pred, y_test):
    print("TRAIN:", list(np.round(history.history['acc'], 4)))
    # print("train_acc =", list(np.round(history.history['acc'], 4))[-1], "\n")
    print("TEST:", list(np.round(history.history['val_acc'], 4)))
    print("LOSS:", list(np.round(history.history['loss'], 4)))
    print("Max F1 was", max(f1_results), "at epoch", f1_results.index(max(f1_results)) + 1)
    print("F1:", f1_results)

    # CONFUSION MATRIX
    # print("confusion matrix:")
    # print(confusion_matrix(y_test, y_pred))


def print_3class_results(history, y_pred, y_test):
    # PRINT FINAL TRAIN/TEST/LOSS INFO
    print("TRAIN:", list(np.round(history.history['acc'], 4)))
    print("TEST:", list(np.round(history.history['val_acc'], 4)))
    print("LOSS:", list(np.round(history.history['loss'], 4)))
    print("\n")

    # MAXIMUMS
    print("Max F1 weighted was", max(f1_results_weighted), "at epoch", f1_results_weighted.index(max(f1_results_weighted)) + 1)
    print("Max F1 micro was", max(f1_results_micro), "at epoch", f1_results_micro.index(max(f1_results_micro)) + 1)

    # CONFUSION MATRIX
    # print("confusion matrix:")
    # print(confusion_matrix(y_test, y_pred))


def test_3_class(filename="cleaned_tweets_16k_3class.csv"):
    print("\nSIMPLE GLOVE MODEL")
    # 0=none, 1=racism, 2=sexism

    # get the data
    X, labels = get_data(filename=filename)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(texts=X)
    vocab_size = len(t.word_index) + 1
    print("VOCAB SIZE =", vocab_size)

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(texts=X)

    # pad documents
    max_len = get_pad_length(filename)
    print(max_len)
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.10)

    labels_train = np_utils.to_categorical(y_train)
    labels_test = np_utils.to_categorical(y_test)

    # load a pre-saved model
    # model = load_model(save_path)

    # embedding_matrix = get_glove_matrix(vocab_size, t)
    embedding_matrix = get_glove_matrix_from_dump()

    # load the model
    model = load_model(LOAD_PATH)

    class_weight = {0: 1.0, 1: 1.0, 2: 1.0}
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # Evaluate on test set
    print("Evaluating...")
    y_prob = model.predict(x=X_test)
    labels_pred = y_prob.argmax(axis=-1)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTEST_ACC = " + str(round(accuracy * 100, 2)) + "%")
    print("Micro Precision = ", round(precision_score(y_test, labels_pred, average='micro'), 4))
    print("Micro Recall = ", round(recall_score(y_test, labels_pred, average='micro'), 4))
    print("Micro F1 = ", round(f1_score(y_test, labels_pred, average='micro'), 4), "\n\n")

    if CONTINUE_TRAIN:
        print("Continuing to train!")
        history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                            nb_epoch=3, batch_size=128, callbacks=[three_class_metrics], class_weight=class_weight)

        y_prob = model.predict(x=X_test)
        labels_pred = y_prob.argmax(axis=-1)
        loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
        print("\bTEST_ACC = " + str(round(accuracy * 100, 2)) + "%")
        print("Micro Precision = ", round(precision_score(y_test, labels_pred, average='micro'), 4))
        print("Micro Recall = ", round(recall_score(y_test, labels_pred, average='micro'), 4))
        print("Micro F1 = ", round(f1_score(y_test, labels_pred, average='micro'), 4), "\n\n")

        print_3class_results(history, labels_pred, y_test)
        print("BEST CONFUSION MATRIX:")
        print(best_confusion_matrix)
        draw_graph(history, classes=3)


def test_2_class(filename="cleaned_dixon.csv"):
    print("\nSIMPLE GLOVE MODEL")

    # get the data
    X, labels = get_data(filename=filename)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(texts=X)
    vocab_size = len(t.word_index) + 1
    print("VOCAB SIZE =", vocab_size)

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(texts=X)

    # pad documents
    max_len = get_pad_length(filename)
    print(max_len)
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # Split into training and test data
    X_train, X_test, labels_train, labels_test = train_test_split(padded_docs, labels, test_size=0.10, random_state=RANDOM_STATE)

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # embedding_matrix = get_glove_matrix(vocab_size, t)
    embedding_matrix = get_glove_matrix_from_dump()

    # load the model
    model = load_model(LOAD_PATH)

    # class_weight = {0: 1.0, 1: 1.0}
    # my_adam = optimizers.Adam(lr=0.003, decay=0.001)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # evaluate on test set
    print("\n\nEvaluating...")
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    y_pred = model.predict(x=X_test)
    y_pred = np.round(y_pred, 0)
    print("Test accuracy = ", round(accuracy, 4))
    print("Precision = ", round(precision_score(labels_test, y_pred), 4))
    print("Recall = ", round(recall_score(labels_test, y_pred), 4))
    print("F1 = ", round(f1_score(labels_test, y_pred), 4), "\n\n")

    # if we want to continue training, run more epochs, evaluate on test and print training graph
    if CONTINUE_TRAIN:
        print("Continuing to train!")
        history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                            epochs=10, batch_size=32, callbacks=[metrics], verbose=1)

        loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
        y_pred = model.predict(x=X_test)
        y_pred = np.round(y_pred, 0)
        print("Test accuracy = ", round(accuracy, 4))
        print("Precision = ", round(precision_score(labels_test, y_pred), 4))
        print("Recall = ", round(recall_score(labels_test, y_pred), 4))
        print("F1 = ", round(f1_score(labels_test, y_pred), 4), "\n\n")

        print_results(history, y_pred, labels_test)
        print("BEST CONFUSION MATRIX:")
        print(best_confusion_matrix)
        draw_graph(history, classes=2)


if __name__ == "__main__":
    # FILE NAMES
    matrix = "cleaned_tweets_16k"
    file = matrix + str(".csv")

    # PARAMETERS
    LOAD_PATH = "twitter_2class_LSTM50"
    SAVE_PATH = LOAD_PATH + str("_retrain")
    CONTINUE_TRAIN = False
    RANDOM_STATE = 3

    # ARCHITECTURE
    test_2_class(file)
    # test_3_class(file)
