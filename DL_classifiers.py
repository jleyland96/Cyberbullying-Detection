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


def get_data(n=20000, filename="cleaned_text_messages.csv"):
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
                if line_count-1 < n:
                    label_bullying = int(row[0])
                    text = row[1]

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

    return embedding_matrix


def get_pad_length(filename):
    if filename == "cleaned_text_messages.csv":
        return 40
    elif filename == "cleaned_twitter_dataset.csv":
        return 30
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


def learn_embeddings_model(filename="cleaned_text_messages.csv"):
    print("\nSIMPLE GLOVE MODEL")

    # get the data
    X, labels = get_data(n=20000, filename=filename)

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

    # TODO: Repeat the positives here if I want to

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Dev 1's proportion = " + str(round(np.count_nonzero(labels_dev) / len(labels_dev), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    # ---------------- EDIT HERE ----------------
    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))
    model.add(LSTM(units=500, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # print(model.summary())

    # fit the model
    print("Fitting the model...")
    model.fit(x=X_train, y=labels_train, validation_data=(X_dev, labels_dev),
              epochs=300, batch_size=128, callbacks=[metrics])
    # ---------------- END EDIT ----------------

    # evaluate
    # labels_pred = model.predict_classes(x=X_test)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")


def simple_glove_LSTM_model(filename="cleaned_text_messages.csv"):
    print("\nSIMPLE GLOVE MODEL")

    # get the data
    X, labels = get_data(n=20000, filename=filename)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(texts=X)
    vocab_size = len(t.word_index) + 1
    print("VOCAB SIZE =", vocab_size)

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(texts=X)

    # pad documents
    max_len = get_pad_length(filename)
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # split to get dev data (0.2), then split to get train/test data (0.7 and 0.1)
    X, X_dev, y, labels_dev = train_test_split(padded_docs, labels, test_size=0.20)
    X_train, X_test, labels_train, labels_test = train_test_split(X, y, test_size=0.125)

    # TODO: Repeat the positives here if I want to

    print("Train 1's proportion = " + str(round(np.count_nonzero(labels_train) / len(labels_train), 4)))
    print("Dev 1's proportion = " + str(round(np.count_nonzero(labels_dev) / len(labels_dev), 4)))
    print("Test 1's proportion = " + str(round(np.count_nonzero(labels_test) / len(labels_test), 4)))
    print()

    embedding_matrix = get_glove_matrix(vocab_size, t)

    # ---------------- EDIT HERE ----------------
    # define the model
    model = Sequential()
    e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
                  input_length=max_len, trainable=False)
    model.add(e)
    model.add(LSTM(units=200, dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())
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
    # ------------------ END EDIT ------------------

    # evaluate
    # labels_pred = model.predict_classes(x=X_test)
    loss, accuracy = model.evaluate(x=X_test, y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")


save_path = "TEST"
print("TEST")
filename = "cleaned_text_messages.csv"
# learn_embeddings_model(filename)
simple_glove_LSTM_model(filename)
