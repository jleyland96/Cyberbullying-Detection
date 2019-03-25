from pickle import load
import numpy as np
import matplotlib.pyplot as plt
import csv
import keras.backend as K
import tensorflow as tf
from numpy import array
from numpy import asarray, zeros
from keras.utils import np_utils
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import pickle

# 'global' variable to store sequence of validation accuracies
validation_results = []
f1_results = []
f1_results_weighted = []
f1_results_micro = []
best_confusion_matrix = ""


# load a clean dataset
def get_data(filename="cleaned_tweets_16k.csv"):
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

    print("processed", line_count - 1, "comments\n")
    return X, y


# get glove matrix for this vocab
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


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# Print results at the end of a 2 class run
def print_results(history, y_pred, y_test):
    print("TRAIN:", list(np.round(history.history['acc'], 4)))
    print("train_acc =", list(np.round(history.history['acc'], 4))[-1], "\n")
    print("TEST:", list(np.round(history.history['val_acc'], 4)), "\n")
    print("LOSS:", list(np.round(history.history['loss'], 4)), "\n")

    # Print F1 history (additional if using F1 loss function)
    if loss_fn == "F1":
        val_f1 = list(np.round(history.history['val_f1'], 4))
        print("VAL_F1:", val_f1, "\n")
        print("Max val_f1 was", max(val_f1), "at epoch", val_f1.index(max(val_f1)) + 1, "\n")
        print("TRAIN_F1:", list(np.round(history.history['f1'], 4)))
    else:
        print("Max F1 was", max(f1_results), "at epoch", f1_results.index(max(f1_results)) + 1, "\n")
        print("F1:", f1_results)

    # CONFUSION MATRIX
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def print_3class_results(y_test, y_pred, history):
    # PRINT FINAL TRAIN/TEST/LOSS INFO
    print("TRAIN:", list(np.round(history.history['acc'], 4)), "\n")
    print("TEST:", list(np.round(history.history['val_acc'], 4)), "\n")
    print("LOSS:", list(np.round(history.history['loss'], 4)), "\n")
    print("\n")

    # PRINT FINAL PRECISION, RECALL, F1 INFO
    # print("Weighted precision:", precision_score(y_test, y_pred, average='weighted'))
    # print("Weighted recall:", recall_score(y_test, y_pred, average='weighted'))
    # print("Weighted F1", f1_score(y_test, y_pred, average='weighted'))
    # print("\n")
    # print("Micro precision:", precision_score(y_test, y_pred, average='micro'))
    # print("Micro recall:", recall_score(y_test, y_pred, average='micro'))
    # print("Micro F1", f1_score(y_test, y_pred, average='micro'))
    # print("\n")

    # MAXIMUMS
    print("Max F1 weighted was", max(f1_results_weighted), "at epoch", f1_results_weighted.index(max(f1_results_weighted)) + 1, "\n")
    print("Max F1 micro was", max(f1_results_micro), "at epoch", f1_results_micro.index(max(f1_results_micro)) + 1, "\n")


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)

    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# draw graph at the end of execution
def draw_graph(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(f1_results)
    plt.plot(history.history['loss'])
    plt.title('model accuracy and f1 score')
    plt.ylabel('accuracy/f1/loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'f1', 'loss'], loc='lower right')
    plt.savefig('DL_graphs/' + str(save_path) + ' graph.png')
    plt.show()


# f1 metric
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


# f1 loss function
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


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        global best_confusion_matrix

        val_predict = (np.asarray(
            self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2]]))).round()
        val_targ = self.validation_data[3]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # print("METRICS")
        print("F1       :", _val_f1)
        # print("PRECISION:", _val_precision)
        # print("RECALL   :", _val_recall)
        validation_results.append(round(_val_acc, 3))
        f1_results.append(round(_val_f1, 3))

        # if the current f1 value is bigger than all of the previous f1 scores, save the model
        if len(f1_results) > 1 and _val_f1 > max(f1_results[:-1]):
            print("SAVING NEW MODEL")
            best_confusion_matrix = confusion_matrix(val_targ, val_predict)
            # Save the model for another time
            # save_model(self.model, save_path)

        # Print validation accuracy and f1 scores (so we can plot later)
        # print("\nVAL_ACC:\n", validation_results)
        # print("\n\n")
        # print("F1:\n", f1_results)

        # print()
        # print(confusion_matrix(val_targ, val_predict))
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
        print()
        val_predict = (np.asarray(
            self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2]]))).round()
        val_targ = self.validation_data[3]

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


three_class_metrics = Three_Class_Metrics()


# define model with glove embeddings as inputs
def define_glove_model(length, vocab_size, embedding_matrix, num_classes):
    # channel 1
    inputs1 = Input(shape=(length,))
    e1 = Embedding(input_dim=vocab_size, output_dim=300,
                   embeddings_initializer=Constant(embedding_matrix), input_length=length)
    e1.trainable = False
    embedding1 = e1(inputs1)
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # channel 2
    inputs2 = Input(shape=(length,))
    e2 = Embedding(input_dim=vocab_size, output_dim=300,
                   embeddings_initializer=Constant(embedding_matrix), input_length=length)
    e2.trainable = False
    embedding2 = e2(inputs2)
    conv2 = Conv1D(filters=64, kernel_size=4, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # channel 3
    inputs3 = Input(shape=(length,))
    e3 = Embedding(input_dim=vocab_size, output_dim=300,
                   embeddings_initializer=Constant(embedding_matrix), input_length=length)
    e3.trainable = False
    embedding3 = e3(inputs3)
    conv3 = Conv1D(filters=64, kernel_size=5, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    # merge
    merged = concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = Dense(30, activation='relu')(merged)

    # DEFINE the right model based on number of classes and loss function
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        if loss_fn == "F1":
            model.compile(loss=f1_loss, optimizer='adam', metrics=['acc', f1])
        else:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:  # num_classes=3
        outputs = Dense(3, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define the model
def define_model(length, vocab_size, num_classes):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 300)(inputs1)
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=64, kernel_size=4, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=64, kernel_size=5, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    # merge
    merged = concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = Dense(10, activation='relu')(merged)

    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:  # num_classes=3
        outputs = Dense(3, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def onehot_2class(filename):
    # load training dataset
    trainLines, trainLabels = get_data(filename=filename)

    # create tokenizer and encode data TODO: change length based on dataset
    tokenizer = create_tokenizer(trainLines)
    length = 500
    vocab_size = len(tokenizer.word_index) + 1
    trainX = encode_text(tokenizer, trainLines, length)

    # split the data
    trainX, testX, trainLabels, testLabels = train_test_split(trainX, trainLabels, test_size=0.20)

    # define, fit and save model
    model = define_model(length, vocab_size, num_classes=2)

    # fit model
    # trainX = np.array(trainX)
    # trainLabels = np.array(trainLabels)
    # testX = np.array(testX)
    # testLabels = np.array(testLabels)

    # FIT
    class_weight = {0: 1.0,
                    1: 1.0}
    history = model.fit(x=[trainX, trainX, trainX], y=array(trainLabels),
                        validation_data=([testX, testX, testX], array(testLabels)),
                        nb_epoch=30, batch_size=256, callbacks=[metrics], class_weight=class_weight, verbose=1)

    # Evaluate
    loss, accuracy = model.evaluate(x=[testX, testX, testX], y=testLabels, verbose=1)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")
    y_pred = model.predict(x=[testX, testX, testX])
    y_pred = np.round(y_pred, 0)

    print_results(history, y_pred, testLabels)
    print("BEST:")
    print(best_confusion_matrix)
    # draw_graph(history)


def onehot_3class():
    # load training dataset
    trainLines, labels = get_data(filename="cleaned_tweets_16k_3class.csv")

    # create tokenizer and encode data
    tokenizer = create_tokenizer(trainLines)
    length = 32
    vocab_size = len(tokenizer.word_index) + 1
    trainX = encode_text(tokenizer, trainLines, length)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(trainX, labels, test_size=0.10)

    labels_train = np_utils.to_categorical(y_train)
    labels_test = np_utils.to_categorical(y_test)

    # define, fit and save model
    model = define_model(length, vocab_size, num_classes=3)

    # FIT
    class_weight = {0: 1.0,
                    1: 1.0,
                    2: 1.0}
    history = model.fit(x=[X_train, X_train, X_train], y=labels_train,
                        validation_data=([X_test, X_test, X_test], labels_test),
                        nb_epoch=50, batch_size=64, callbacks=[three_class_metrics], class_weight=class_weight)

    # Evaluate
    loss, accuracy = model.evaluate(x=[X_test, X_test, X_test], y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")
    labels_pred = model.predict(x=[X_test, X_test, X_test])

    print_3class_results(y_test, labels_pred, history)


def glove_2class(filename):
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

    # pad documents TODO: Change max_len based on dataset
    max_len = 500
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.20)

    embedding_matrix = get_glove_matrix(vocab_size, t)
    # embedding_matrix = get_glove_matrix_from_dump()

    # Define our model, taking glove embeddings as inputs and outputting classes
    model = define_glove_model(max_len, vocab_size, embedding_matrix, num_classes=2)

    # FIT
    class_weight = {0: 1.0,
                    1: 1.0,
                    2: 1.0}
    history = model.fit(x=[X_train, X_train, X_train], y=y_train,
                        validation_data=([X_test, X_test, X_test], y_test),
                        nb_epoch=30, batch_size=256, callbacks=[metrics], class_weight=class_weight, verbose=1)

    # Evaluate
    if loss_fn == "F1":
        loss, accuracy, _ = model.evaluate(x=[X_test, X_test, X_test], y=y_test, verbose=1)
    else:
        loss, accuracy = model.evaluate(x=[X_test, X_test, X_test], y=y_test, verbose=1)

    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")
    y_pred = model.predict(x=[X_test, X_test, X_test])
    y_pred = np.round(y_pred, 0)

    print_results(history, y_pred, y_test)
    print("BEST:")
    print(best_confusion_matrix)
    # draw_graph(history)


def glove_3class():
    print("\nSIMPLE GLOVE MODEL")
    # 0=none, 1=racism, 2=sexism

    # get the data
    X, labels = get_data(filename="cleaned_tweets_16k_3class.csv")

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(texts=X)
    vocab_size = len(t.word_index) + 1
    print("VOCAB SIZE =", vocab_size)

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(texts=X)

    # pad documents
    max_len = 32
    padded_docs = pad_sequences(sequences=encoded_docs, maxlen=max_len, padding='post')

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.10)

    labels_train = np_utils.to_categorical(y_train)
    labels_test = np_utils.to_categorical(y_test)

    # embedding_matrix = get_glove_matrix(vocab_size, t)
    embedding_matrix = get_glove_matrix_from_dump()

    model = define_glove_model(max_len, vocab_size, embedding_matrix, num_classes=3)

    # FIT
    class_weight = {0: 1.0,
                    1: 1.0,
                    2: 1.0}
    history = model.fit(x=[X_train, X_train, X_train], y=labels_train,
                        validation_data=([X_test, X_test, X_test], labels_test),
                        nb_epoch=50, batch_size=64, callbacks=[three_class_metrics], class_weight=class_weight)

    # Evaluate
    labels_pred = model.predict(x=[X_test, X_test, X_test])
    # labels_pred = model.predict_classes(x=[X_test, X_test, X_test])
    loss, accuracy = model.evaluate(x=[X_test, X_test, X_test], y=labels_test, verbose=0)
    print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")

    print_3class_results(y_test, labels_pred, history)


save_path = "Testing"
file = 'cleaned_dixon.csv'
loss_fn = "not F1"
glove_2class(filename=file)
# onehot_2class(filename=file)
