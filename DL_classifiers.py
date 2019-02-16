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
from keras.layers import LSTM
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


# 'global' variable to store sequence of validation accuracies
validation_results = []
f1_results = []


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


def repeat_positives(old_x, old_y, repeats=2):
    new_x = []
    new_y = []

    # rebuild the X dataset
    for i in range(len(old_x)):
        new_x.append(old_x[i])
        new_y.append(old_y[i])

        # if the example is a positive examples, repeat it in the dataset
        if old_y[i] == 1:
            for j in range(repeats-1):
                new_x.append(old_x[i])
                new_y.append(old_y[i])

    return new_x, new_y


def count_vocab_size(x):
    vec = TfidfVectorizer()
    corpus_fit_transform = vec.fit_transform(x)
    return corpus_fit_transform.shape[1]


def shuffle_data(X, y):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    return X, y


def get_glove_matrix_from_dump():
    embedding_matrix = pickle.load(open('embedding_matrix.p', 'rb'))
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

    pickle.dump(embedding_matrix, open('embedding_matrix.p', 'wb'), protocol=2)

    return embedding_matrix


def get_pad_length(filename):
    if filename == "cleaned_text_messages.csv":
        return 32
    elif filename == "cleaned_twitter_1K.csv":
        return 30
    elif filename == "cleaned_tweets_16k.csv" or filename == "cleaned_tweets_16k_3class.csv":
        return 32
    else:  # cleaned_formspring.csv is up to length 1200
        return 100


def save_model(model, path):
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


metrics = Metrics()


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


def print_results(history):
    print("TRAIN:", list(np.round(history.history['acc'], 4)), "\n")
    print("TEST:", list(np.round(history.history['val_acc'], 4)), "\n")
    if loss == "F1":
        val_f1 = list(np.round(history.history['val_f1'], 4))
        print("VAL_F1:", val_f1, "\n")
        print("Max val_f1 was", max(val_f1), "at epoch", val_f1.index(max(val_f1)) + 1, "\n")
        print("TRAIN_F1:", list(np.round(history.history['f1'], 4)))
    else:
        print("Max F1 was", max(f1_results), "at epoch", f1_results.index(max(f1_results)) + 1, "\n")
        print("F1:", f1_results)


def learn_embeddings_model(filename="cleaned_text_messages.csv"):
    print("\nLEARN EMBEDDINGS MODEL")

    # get the data
    X, labels = get_data(filename=filename)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(texts=X)
    vocab_size = len(t.word_index) + 1
    print("VOCAB SIZE =", vocab_size)

    # integer encode the documents
    encoded_docs = [one_hot(x, vocab_size) for x in X]

    # pad documents
    max_len = get_pad_length(filename)
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # split to get dev data (0.2), then split to get train/test data (0.7 and 0.1)
    X, X_dev, y, labels_dev = train_test_split(padded_docs, labels, test_size=0.20)
    X_train, X_test, labels_train, labels_test = train_test_split(X, y, test_size=0.125)

    # Repeat the positives here if I want to
    # X_train, labels_train = repeat_positives(X_train, labels_train, repeats=8)

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Dev 1's proportion = " + str(round(np.count_nonzero(labels_dev) / len(labels_dev), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # ---------------- EDIT LEARN EMBEDDINGS HERE ----------------
    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))
    model.add(LSTM(units=200, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # print(model.summary())

    # fit the model
    print("Fitting the model...")
    model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_dev, labels_dev),
              epochs=300, batch_size=128, callbacks=[metrics])
    # ---------------- END LEARN EMBEDDINGS EDIT ----------------

    # evaluate
    # labels_pred = model.predict_classes(x=X_test)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")


def dense_network(model):
    model.add(Flatten())
    model.add(Dense(units=20, activation=None))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=1, activation='sigmoid'))
    return model


def cnn_lstm_network(model):
    model.add(Conv1D(filters=64, kernel_size=3, strides=2, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())

    model.add(Dense(units=1, activation='sigmoid'))
    return model


def cnn_network(model):
    model.add(Conv1D(filters=32, kernel_size=4, strides=2, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Conv1D(filters=48, kernel_size=3, strides=2, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    return model


def main_3_class_model(filename="cleaned_tweets_16k_3class.csv"):
    print("\nSIMPLE GLOVE MODEL")
    # 0=none, 1=racism, 2=sexism

    # get the data
    X, labels = get_data(filename=filename)
    print(labels[:50])
    category_labels = np_utils.to_categorical(labels)

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
    X_train, X_test, labels_train, labels_test = train_test_split(padded_docs, category_labels, test_size=0.10)

    # load a pre-saved model
    # model = load_model(save_path)

    # embedding_matrix = get_glove_matrix(vocab_size, t)
    embedding_matrix = get_glove_matrix_from_dump()

    # ---------------- MODEL HERE ----------------
    # Embedding input
    model = Sequential()
    # e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
    #               input_length=max_len, trainable=False)
    e = Embedding(input_dim=vocab_size, output_dim=300,
                  embeddings_initializer=Constant(embedding_matrix), input_length=max_len)
    e.trainable = False
    model.add(e)

    model.add(LSTM(units=100, recurrent_dropout=0.5, dropout=0.5))

    model.add(Dense(units=3, activation='softmax'))

    # compile the model
    # adam = optimizers.Adam(lr=0.0005, decay=0.01, beta_1=0.92, beta_2=0.9992)
    # print("F1 LOSS")
    # model.compile(optimizer='adam', loss=f1_loss, metrics=['acc', f1])
    # history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
    #                     nb_epoch=30, batch_size=128, class_weight=class_weight)

    class_weight = {0: 1.0,
                    1: 1.0,
                    2: 1.0}
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                        nb_epoch=50, batch_size=256, class_weight=class_weight)
    # ------------------ END MODEL ------------------

    # evaluate
    labels_pred = model.predict_classes(x=X_test)
    print(labels_pred)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")


def main_2_class_model(filename="cleaned_text_messages.csv"):
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
    X_train, X_test, labels_train, labels_test = train_test_split(padded_docs, labels, test_size=0.10)

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # load a pre-saved model
    # model = load_model(save_path)

    # embedding_matrix = get_glove_matrix(vocab_size, t)
    embedding_matrix = get_glove_matrix_from_dump()

    # GloVe hit rate
    print(np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1)) / vocab_size)

    # ---------------- MODEL HERE ----------------
    # Embedding input
    model = Sequential()
    # e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
    #               input_length=max_len, trainable=False)
    e = Embedding(input_dim=vocab_size, output_dim=300,
                  embeddings_initializer=Constant(embedding_matrix), input_length=max_len)
    e.trainable = False
    model.add(e)

    model.add(LSTM(units=100, recurrent_dropout=0.5))

    model.add(Dense(units=1, activation='sigmoid'))

    # compile the model
    # adam = optimizers.Adam(lr=0.0005, decay=0.01, beta_1=0.92, beta_2=0.9992)
    # print("F1 LOSS")
    # model.compile(optimizer='adam', loss=f1_loss, metrics=['acc', f1])
    # history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
    #                     nb_epoch=30, batch_size=128, class_weight=class_weight)

    class_weight = {0: 1.0,
                    1: 1.0}
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                        nb_epoch=150, batch_size=256, callbacks=[metrics], class_weight=class_weight)
    # ------------------ END MODEL ------------------

    # evaluate
    labels_pred = model.predict_classes(x=X_test)
    print(labels_pred)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")

    print_results(history)

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.plot(f1_results)
    # plt.title('model accuracy and f1 score')
    # plt.ylabel('accuracy/f1')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test', 'f1'], loc='upper left')
    # plt.savefig("keras_vis/5000 graph.png")


save_path = "TEST"
loss = "cross-entropy"
file = "cleaned_tweets_16k.csv"
main_2_class_model(file)
# main_3_class_model(file)
