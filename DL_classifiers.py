import csv
import os
import re
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


# 'global' variable to store sequence of validation accuracies
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
            save_model(self.model)

        print(confusion_matrix(val_targ, val_predict))
        print("\n")
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
        print("")
        # print("F1 weighted:\n", f1_results_weighted)
        print("F1 micro:\n", f1_results_micro)

        if (len(f1_results_micro) > 1 and _val_f1_micro > max(f1_results_micro[:-1])) or (len(f1_results_micro) == 1):
            val_targ = val_targ.argmax(axis=-1)
            val_predict = val_predict.argmax(axis=-1)
            best_confusion_matrix = confusion_matrix(val_targ, val_predict)

            print("SAVING NEW MODEL")
            # Save the model for another time
            save_model(self.model)

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


def get_test_size(filename):
    if filename == "cleaned_dixon.csv":
        return 0.20
    else:
        return 0.10


def save_model(model):
    if not(file == "cleaned_dixon.csv"):
        # serialize model to JSON
        model_json = model.to_json()
        with open("saved_models/" + str(SAVE_PATH) + ".json", "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("saved_models/" + str(SAVE_PATH) + ".h5")
    print("Saved " + str(SAVE_PATH) + " model to disk")


def load_model():
    # load json and create model
    json_file = open("saved_models/" + str(LOAD_PATH) + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/" + str(LOAD_PATH) + ".h5")
    print("Loaded " + str(LOAD_PATH) + " model from disk")
    return loaded_model


def draw_graph(history, num_classes):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    if num_classes == 2:
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


def print_results(history, y_pred, y_test):
    print("TRAIN:", list(np.round(history.history['acc'], 4)))
    # print("train acc =", list(np.round(history.history['acc'], 4))[-1])
    print("TEST:", list(np.round(history.history['val_acc'], 4)))
    print("LOSS:", list(np.round(history.history['loss'], 4)))
    if loss == "F1":
        val_f1 = list(np.round(history.history['val_f1'], 4))
        print("VAL_F1:", val_f1, "\n")
        print("Max val_f1 was", max(val_f1), "at epoch", val_f1.index(max(val_f1)) + 1, "\n")
        print("TRAIN_F1:", list(np.round(history.history['f1'], 4)))
    else:
        print("F1:", f1_results, "\n")
        print("Max F1 was", max(f1_results), "at epoch", f1_results.index(max(f1_results)) + 1)

    # CONFUSION MATRIX
    # print("confusion matrix:")
    # print(confusion_matrix(y_test, y_pred))


def print_3class_results(history, y_pred, y_test):
    # PRINT FINAL TRAIN/TEST/LOSS INFO
    print("TRAIN:", list(np.round(history.history['acc'], 4)), "\n")
    print("TEST:", list(np.round(history.history['val_acc'], 4)), "\n")
    print("LOSS:", list(np.round(history.history['loss'], 4)), "\n")
    print("\n")

    # MAXIMUMS
    print("Max F1 weighted was", max(f1_results_weighted), "at epoch", f1_results_weighted.index(max(f1_results_weighted)) + 1, "\n")
    print("Max F1 micro was", max(f1_results_micro), "at epoch", f1_results_micro.index(max(f1_results_micro)) + 1, "\n")

    # CONFUSION MATRIX
    # print("confusion matrix:")
    # print(confusion_matrix(y_test, y_pred))


def clean_message(message, data_choice):
    if data_choice == "1":
        message = re.sub('[\":=#&;\'?!,./\\\*\\n]', '', message.lower())  # remove punctuation
    elif data_choice == "4":
        message = re.sub(r'http\S+', '', message)  # remove URLs
        message = re.sub('[\":=#&;\'?!@,./\\\\\n*]', '', message)  # remove punctuation
        message = re.sub(' +', ' ', message.lower())  # remove multiple spaces, replace with one space
    else:
        message = re.sub('RT @[\w_]+ ', '', message)  # remove RT tags. was RT @[\w_]+:
        message = re.sub('@[\w_]+', '', message)  # remove mentions
        message = re.sub(r'http\S+', '', message)  # remove URLs
        message = re.sub(' +', ' ', message)  # remove multiple spaces, replace with one space
        message = re.sub('[\":=#&;\'?!@,./\\\\\n*]', '', message.lower())  # remove punctuation

    print("Cleaned message:", message)
    return message


def learn_embeddings_2class_f1_loss(filename="cleaned_tweets_16k.csv"):
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
    print(max_len)
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # split to get dev data (0.2), then split to get train/test data (0.7 and 0.1)
    X_train, X_test, labels_train, labels_test = train_test_split(padded_docs, labels, test_size=0.10)

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))

    model.add(LSTM(units=50, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(units=1, activation='sigmoid'))
    # compile the model

    # my_adam = optimizers.Adam(lr=0.005, decay=0.05)
    model.compile(optimizer='adam', loss=f1_loss, metrics=['acc', f1])
    print(model.summary())

    # fit the model
    print("Fitting the model...")
    history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                        epochs=NUM_EPOCHS, batch_size=128, callbacks=[metrics])

    # evaluate
    y_pred = model.predict(x=X_test)
    y_pred = np.round(y_pred, 0)
    print_results(history, y_pred, labels_test)
    draw_graph(history, num_classes=2)
    print("Best confusion matrix:")
    print(best_confusion_matrix)


def learn_embeddings_model_2class(filename="cleaned_tweets_16k.csv"):
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
    X_train, X_test, labels_train, labels_test = train_test_split(padded_docs, labels, test_size=0.10)

    # Repeat the positives here if I want to
    # X_train, labels_train = repeat_positives(X_train, labels_train, repeats=8)

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # ---------------- EDIT LEARN EMBEDDINGS HERE ----------------
    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=max_len))

    model.add(LSTM(units=50, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(units=1, activation='sigmoid'))
    # compile the model

    # my_adam = optimizers.Adam(lr=0.005, decay=0.05)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    # fit the model
    print("Fitting the model...")
    class_weight = {0: 1.0, 1: 1.0}
    history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                        epochs=NUM_EPOCHS, batch_size=128, callbacks=[metrics])
    # ---------------- END LEARN EMBEDDINGS EDIT ----------------

    # evaluate
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    y_pred = model.predict(x=X_test)
    y_pred = np.round(y_pred, 0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")
    print_results(history, y_pred, labels_test)
    print(draw_graph(history, num_classes=2))
    print("Best confusion matrix:")
    print(best_confusion_matrix)


def learn_embeddings_model_3class(filename="cleaned_tweets_16k_3class.csv"):
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
    X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=TEST_SIZE)
    labels_train = np_utils.to_categorical(y_train)
    labels_test = np_utils.to_categorical(y_test)

    print("Train 1's proportion = " + str(round(np.count_nonzero(y_train) / len(y_train), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(y_test) / len(y_test), 4)))
    print()

    # ---------------- EDIT LEARN EMBEDDINGS HERE ----------------
    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))

    model.add(LSTM(units=50, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(units=3, activation='softmax'))
    # compile the model

    # my_adam = optimizers.Adam(lr=0.005, decay=0.05)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    # fit the model
    print("Fitting the model...")
    class_weight = {0: 1.0, 1: 1.0}
    history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                        epochs=NUM_EPOCHS, batch_size=128, callbacks=[three_class_metrics])
    # ---------------- END LEARN EMBEDDINGS EDIT ----------------

    # evaluate
    y_prob = model.predict(x=X_test)
    labels_pred = y_prob.argmax(axis=-1)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTEST_ACC = " + str(round(accuracy * 100, 2)) + "%")

    print_3class_results(history, labels_pred, y_test)
    draw_graph(history, num_classes=3)
    print("Best confusion matrix:")
    print(best_confusion_matrix)


def dense_network(model):
    model.add(Flatten())
    model.add(Dense(units=20, activation=None))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(rate=0.4))
    return model


def cnn_lstm_network(model):
    model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))
    return model


def cnn_network(model):
    model.add(Conv1D(filters=32, kernel_size=4, strides=2, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='valid', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Flatten())
    return model


def main_2_class_f1_loss(filename="cleaned_twitter_1K.csv"):
    print("\nGLOVE MODEL")

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

    # ---------------- MODEL HERE ----------------
    # Embedding input
    model = Sequential()
    # e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
    #               input_length=max_len, trainable=False)
    e = Embedding(input_dim=vocab_size, output_dim=300,
                  embeddings_initializer=Constant(embedding_matrix), input_length=max_len)
    e.trainable = False
    model.add(e)

    model.add(LSTM(units=50, dropout=0.3, recurrent_dropout=0.3))
    # model.add(BatchNormalization())
    # model.add(Bidirectional(LSTM(units=400, dropout=0.5, recurrent_dropout=0.5)))

    # model = cnn_lstm_network(model)

    model.add(Dense(units=1, activation='sigmoid'))

    # compile the model
    # adam = optimizers.Adam(lr=0.0005, decay=0.01, beta_1=0.92, beta_2=0.9992)
    print("F1 LOSS")
    model.compile(optimizer='adam', loss=f1_loss, metrics=['acc', f1])
    print(model.summary())
    history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                        epochs=NUM_EPOCHS, callbacks=[metrics], batch_size=128)

    # evaluate
    # loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    loss, accuracy, _ = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")

    y_pred = model.predict(x=X_test)
    y_pred = np.round(y_pred, 0)
    print_results(history, y_pred, labels_test)
    print("BEST:")
    print(best_confusion_matrix)
    draw_graph(history, num_classes=2)


def main_3_class_model(filename="cleaned_tweets_16k_3class.csv"):
    global NUM_EPOCHS
    print("\nGLOVE MODEL")
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
    X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    labels_train = np_utils.to_categorical(y_train)
    labels_test = np_utils.to_categorical(y_test)

    # load a pre-saved model
    # model = load_model(save_path)

    # embedding_matrix = get_glove_matrix(vocab_size, t)
    embedding_matrix = get_glove_matrix_from_dump()

    # if we want to load the model or not
    if LOAD_MODEL:
        model = load_model()
    else:
        model = Sequential()
        e = Embedding(input_dim=vocab_size, output_dim=300,
                      embeddings_initializer=Constant(embedding_matrix), input_length=max_len)
        e.trainable = False  # should be false
        model.add(e)

        if ARCH_CHOICE == "1":
            model.add(LSTM(units=100, dropout=0.5, recurrent_dropout=0.5))
        elif ARCH_CHOICE == "2":
            model = cnn_network(model)
        elif ARCH_CHOICE == "3":
            model = cnn_lstm_network(model)
        else:
            model.add(Bidirectional(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5)))

        model.add(Dense(units=3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    global CONTINUE_TRAINING
    if not CONTINUE_TRAINING:
        y_prob = model.predict(x=X_test)
        labels_pred = y_prob.argmax(axis=-1)
        loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
        print("\bTEST_ACC = " + str(round(accuracy * 100, 2)) + "%")
        print("Micro Precision = ", round(precision_score(y_test, labels_pred, average='micro'), 4))
        print("Micro Recall = ", round(recall_score(y_test, labels_pred, average='micro'), 4))
        print("Micro F1 = ", round(f1_score(y_test, labels_pred, average='micro'), 4), "\n")
        print("Confusion matrix:\n", confusion_matrix(y_test, labels_pred))

        print("\nWould you like to continue training? ('y' or 'n')")
        inp = input()
        if inp == 'y':
            CONTINUE_TRAINING = True
            NUM_EPOCHS = 3

    # If we want to continue training
    if CONTINUE_TRAINING:
        class_weight = {0: 1.0, 1: 1.0, 2: 1.0}
        history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                            epochs=NUM_EPOCHS, batch_size=128, callbacks=[three_class_metrics], class_weight=class_weight)

        # evaluate
        # labels_pred = model.predict_classes(x=X_test)
        y_prob = model.predict(x=X_test)
        labels_pred = y_prob.argmax(axis=-1)
        loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
        print("\bTEST_ACC = " + str(round(accuracy * 100, 2)) + "%")

        print_3class_results(history, labels_pred, y_test)
        print("Best confusion matrix:")
        print(best_confusion_matrix)
        draw_graph(history, num_classes=3)

    return t, model, max_len


def main_2_class_model(filename="cleaned_dixon.csv"):
    print("\nGLOVE MODEL")
    global NUM_EPOCHS

    # get the data
    X, labels = get_data(filename=filename)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(texts=X)
    vocab_size = len(t.word_index) + 1
    print("VOCAB SIZE =", vocab_size)

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(texts=X)
    # for i in range(0, 3):
    #     print(X[i])
    #     print(encoded_docs[i])
    # print(t.texts_to_sequences(texts=["hello what is going on jansdiandisnai here usddisadisinosadmop"]))

    # pad documents
    # max_len = 100
    max_len = get_pad_length(filename)
    print("pad length =" + str(max_len))
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # Split into training and test data
    X_train, X_test, labels_train, labels_test = train_test_split(padded_docs, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # embedding_matrix = get_glove_matrix(vocab_size, t)
    embedding_matrix = get_glove_matrix_from_dump()

    if LOAD_MODEL:
        # load a pre-saved model
        if file == "cleaned_dixon.csv":
            # construct the model and just load the weights
            model = Sequential()
            e = Embedding(input_dim=vocab_size, output_dim=300, embeddings_initializer=Constant(embedding_matrix), input_length=max_len)
            e.trainable = False
            model.add(e)
            model.add(LSTM(units=500, dropout=0.5, recurrent_dropout=0.5))
            model.add(Dense(units=1, activation='sigmoid'))
        else:
            model = load_model()
    else:
        # ---------------- MODEL HERE ----------------
        # Embedding input
        model = Sequential()
        # e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
        #               input_length=max_len, trainable=False)
        e = Embedding(input_dim=vocab_size, output_dim=300,
                      embeddings_initializer=Constant(embedding_matrix), input_length=max_len)
        e.trainable = False
        model.add(e)

        if ARCH_CHOICE == "1":
            model.add(LSTM(units=100, dropout=0.5, recurrent_dropout=0.5))
        elif ARCH_CHOICE == "2":
            model = cnn_network(model)
        elif ARCH_CHOICE == "3":
            model = cnn_lstm_network(model)
        elif ARCH_CHOICE == "4":
            model.add(Bidirectional(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5)))
        else:
            model.add(LSTM(units=400, dropout=0.4, recurrent_dropout=0.4))
        model.add(Dense(units=1, activation='sigmoid'))

    # class_weight = {0: 1.0, 1: 1.0}
    # my_adam = optimizers.Adam(lr=0.003, decay=0.001)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    global CONTINUE_TRAINING
    if not CONTINUE_TRAINING:
        # Evaluate
        loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
        y_pred = model.predict(x=X_test)
        y_pred = np.round(y_pred, 0)
        print("Test accuracy = ", round(accuracy, 4))
        print("Precision = ", round(precision_score(labels_test, y_pred), 4))
        print("Recall = ", round(recall_score(labels_test, y_pred), 4))
        print("F1 = ", round(f1_score(labels_test, y_pred), 4), "\n\n")

        print("Would you like to continue training? ('y' or 'n')")
        inp = input()
        if inp == 'y':
            CONTINUE_TRAINING = True
            NUM_EPOCHS = 3

    if CONTINUE_TRAINING:
        history = model.fit(x=np.array(X_train), y=np.array(labels_train), validation_data=(X_test, labels_test),
                            epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[metrics], verbose=1)
        # ------------------ END MODEL ------------------

        # evaluate
        loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
        y_pred = model.predict(x=X_test)
        y_pred = np.round(y_pred, 0)
        print("\bTest accuracy = " + str(round(accuracy, 4)))

        print_results(history, y_pred, labels_test)
        print("Best confusion matrix:")
        print(best_confusion_matrix)
        draw_graph(history, num_classes=2)

    return t, model, max_len


def main_menu():
    global loss
    global matrix
    global file
    global RANDOM_STATE
    global LOAD_MODEL
    global SAVE_PATH
    global TEST_SIZE
    global LOAD_PATH
    global CONTINUE_TRAINING
    global ARCH_CHOICE
    global NUM_EPOCHS
    global BATCH_SIZE

    print("  MENU\n--------")

    # --- DATASET ---
    print("Choose a dataset: ('1', '2', '3' or '4')")
    print("1. Twitter_small")
    print("2. Twitter_big_2class")
    print("3. Twitter_big_3class")
    print("4. Reddit")
    data_choice = input()
    if data_choice == "1":
        print("Twitter_small")
        matrix = "cleaned_twitter_1K"
        NUM_EPOCHS = 10
    elif data_choice == "2":
        print("Twitter_big_2class")
        matrix = "cleaned_tweets_16k"
        NUM_EPOCHS = 3
    elif data_choice == "3":
        print("Twitter_big_3class")
        matrix = "cleaned_tweets_16k_3class"
        NUM_EPOCHS = 3
    else:
        print("Reddit")
        matrix = "cleaned_dixon"
        NUM_EPOCHS = 1
        BATCH_SIZE = 256
    file = matrix + str(".csv")
    TEST_SIZE = get_test_size(file)
    loss = "not F1"
    print(matrix)
    print(file)

    # --- LOAD or TRAIN? ---
    print("Observe a saved model ('1'), or train a new model ('2')?")
    choice = input()
    if choice == "1":
        # Loading a saved model
        LOAD_MODEL = True
        CONTINUE_TRAINING = False
        SAVE_PATH = "DEMO SAVE PATH"
        if data_choice == "1":
            print("Loading LSTM model on twitter_small")
            LOAD_PATH = "twitter_small_LSTM150"
            RANDOM_STATE = 2
            t, model, max_len = main_2_class_model(file)
        elif data_choice == "2":
            print("Loading LSTM model on twitter_big_2class")
            LOAD_PATH = "twitter_2class_LSTM50"
            RANDOM_STATE = 3
            t, model, max_len = main_2_class_model(file)
        elif data_choice == "3":
            print("Loading Bidirectional LSTM model on twitter_big_3class")
            LOAD_PATH = "twitter_3class_BI-LSTM"
            RANDOM_STATE = 2
            t, model, max_len = main_3_class_model(file)
        else:
            print("Loading LSTM model on dixon dataset")
            LOAD_PATH = ""  # TODO: fill in dixon data
            RANDOM_STATE = 2
            t, model, max_len = main_2_class_model(file)
    else:
        # training a new model
        LOAD_MODEL = False
        CONTINUE_TRAINING = True
        RANDOM_STATE = 5
        SAVE_PATH = "DEMO TRAIN SAVE PATH"
        print("Which model would you like to try? ('1', '2', '3' or '4')")
        print("1. LSTM")
        print("2. CNN")
        print("3. Combo (CNN + LSTM)")
        print("4. Bidirectional LSTM")
        ARCH_CHOICE = input()

        if data_choice in ["1", "2", "4"]:
            t, model, max_len = main_2_class_model(file)
        else:
            t, model, max_len = main_3_class_model(file)

    # Interact with model and classify the results
    print("\nType a message to interact with the model, or type 'q' to exit")
    message = input()
    while message != "q":
        # Pre-process the input
        message = clean_message(message, data_choice)
        message = t.texts_to_sequences(texts=[message])
        print("\ninteger encoded:", message)
        message = pad_sequences(sequences=message, maxlen=max_len, padding='post')
        print("padded:", message, "\n")

        # make a prediction
        y_pred = model.predict(x=message)
        if data_choice == "3":
            print("Racism prob=" + str(y_pred[0][1]) + ". Sexism prob=" + str(y_pred[0][2]) + ". Neither prob:" + str(y_pred[0][0]))
        else:
            print("Cyberbullying probability:", y_pred[0][0])

        # print the prediction
        if data_choice in ["1", "2", "4"]:
            y_pred = np.round(y_pred, 0)
            if y_pred[0][0] == 0:
                print("Not Cyberbullying")
            else:
                print("Cyberbullying")
        else:
            y_pred = y_pred.argmax(axis=-1)
            if y_pred[0] == 0:
                print("Neither")
            elif y_pred[0] == 1:
                print("Racism")
            else:
                print("Sexism")

        # loop for another test message again?
        print("\ntype message to interact with model, or type 'q' to exit")
        message = input()


if __name__ == "__main__":
    # FILE NAMES
    matrix = "cleaned_dixon"
    file = matrix + str(".csv")
    LOAD_PATH = "dixon_LSTM400"
    SAVE_PATH = LOAD_PATH + str("_TEST")

    # PARAMETERS
    LOAD_MODEL = False
    CONTINUE_TRAINING = True
    RANDOM_STATE = 2
    NUM_EPOCHS = 30
    loss = "not F1"

    # Ignore these
    ARCH_CHOICE = 5
    BATCH_SIZE = 256
    TEST_SIZE = get_test_size(file)

    # main_menu()

    # ARCHITECTURE  (if f1 loss, 16k tweets best. learn_embeddings_f1_loss poor))
    # learn_embeddings_model_2class(file)
    # learn_embeddings_model_3class(filename="cleaned_tweets_16k_3class.csv")
    # learn_embeddings_2class_f1_loss(file)
    # main_2_class_f1_loss(file)
    main_2_class_model(file)
    # main_3_class_model(filename="cleaned_tweets_16k_3class.csv)
