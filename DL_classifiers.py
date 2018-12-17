import csv
import os
import numpy as np
import random
from numpy import asarray, zeros
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers.embeddings import Embedding
from keras.models import model_from_json
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


# 'global' variable to store sequence of validation accuracies
validation_results = []
f1_results = []


def get_data(n=69523):
    X = []
    y = []

    with open('cleaned_dixon_train_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:

            if line_count == 0:
                pass
            else:
                if line_count-1 < n:
                    attack_0 = int(row[6])
                    comment = row[1]

                    X.append(comment)
                    y.append(attack_0)

            line_count += 1

    print("processed", line_count-1, "comments\n")
    return X, y


def count_vocab_size(x):
    vec = TfidfVectorizer()
    corpus_fit_transform = vec.fit_transform(x)
    return corpus_fit_transform.shape[1]


def shuffle_data(X, y):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    return X, y


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
        validation_results.append(round(_val_acc, 3))
        f1_results.append(round(_val_f1, 3))

        # Print validation accuracy and f1 scores (so we can plot later)
        print("\nVAL_ACC:\n", validation_results)
        print("\n\n")
        print("F1:\n", f1_results)

        # Save the model for another time
        # save_model(self.model, save_path)
        return


metrics = Metrics()


def simple_one_hot_model():
    # n = 1000, max_len = 500, output_dim = 64, units = 10, activation = 'relu', epochs = 50
    # |  accuracy = 67.0%
    print("\nSIMPLE ONE-HOT MODEL")

    # get the data
    X, labels = get_data(n=1000)
    vocab_size = count_vocab_size(X)
    print("Vocab size =", vocab_size)

    # pre-process the data
    max_len = 500
    encoded_docs = [one_hot(text=x, n=vocab_size) for x in X]
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # split the data
    X_train, X_test, labels_train, labels_test = train_test_split(padded_docs, labels, test_size=0.20)

    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # print(model.summary())

    # fit the model
    print("fitting the model...")
    model.fit(x=X_train, y=labels_train, validation_data=(X_test, labels_test), nb_epoch=30, batch_size=128, callbacks=[metrics])

    # evaluate
    labels_pred = model.predict_classes(x=X_test)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("Test accuracy = " + str(round(accuracy*100, 2)) + "%")


def learn_embeddings_model():
    print("\nSIMPLE GLOVE MODEL")

    # get the data
    X, labels = get_data(n=69523)
    X, labels = shuffle_data(X, labels)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(texts=X)
    vocab_size = len(t.word_index) + 1
    print("VOCAB SIZE =", vocab_size)

    # integer encode the documents
    encoded_docs = [one_hot(x, vocab_size) for x in X]

    # pad documents
    max_len = 500
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # split to get dev data (0.2)
    X, X_dev, y, labels_dev = train_test_split(padded_docs, labels, test_size=0.20)
    # split to get train/test data (0.7 and 0.1)
    X_train, X_test, labels_train, labels_test = train_test_split(X, y, test_size=0.125)

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Dev 1's proportion = " + str(round(np.count_nonzero(labels_dev) / len(labels_dev), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=max_len))
    model.add(LSTM(units=500, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # print(model.summary())

    # fit the model
    print("Fitting the model...")
    model.fit(x=X_train, y=labels_train, validation_data=(X_dev, labels_dev),
              epochs=40, batch_size=128, callbacks=[metrics])

    # evaluate
    # labels_pred = model.predict_classes(x=X_test)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")


def simple_glove_LSTM_model():
    print("\nSIMPLE GLOVE MODEL")

    # get the data
    X, labels = get_data(n=69523)
    X, labels = shuffle_data(X, labels)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(texts=X)
    vocab_size = len(t.word_index) + 1
    print("VOCAB SIZE =", vocab_size)

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(texts=X)

    # pad documents
    max_len = 500
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # split to get dev data (0.2)
    X, X_dev, y, labels_dev = train_test_split(padded_docs, labels, test_size=0.20)
    # split to get train/test data (0.7 and 0.1)
    X_train, X_test, labels_train, labels_test = train_test_split(X, y, test_size=0.125)

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Dev 1's proportion = " + str(round(np.count_nonzero(labels_dev) / len(labels_dev), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # load embeddings into memory
    glove_vector_size = 300
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

    # define the model
    model = Sequential()
    e = Embedding(input_dim=vocab_size, output_dim=glove_vector_size, weights=[embedding_matrix],
                  input_length=max_len, trainable=False)
    model.add(e)
    model.add(Conv1D(filters=30, kernel_size=3, strides=2, padding='valid'))
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))
    model.add(LSTM(units=200, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    # load a pre-saved model
    # model = load_model(save_path)

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # print(model.summary())

    # fit the model
    print("Fitting the model...")
    model.fit(x=X_train, y=labels_train, validation_data=(X_dev, labels_dev),
              nb_epoch=300, batch_size=128, callbacks=[metrics])

    # evaluate
    # labels_pred = model.predict_classes(x=X_test)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")


print("Final dropout test")
simple_glove_LSTM_model()