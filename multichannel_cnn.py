from pickle import load
import numpy as np
import csv
from numpy import array
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


# 'global' variable to store sequence of validation accuracies
validation_results = []
f1_results = []


# load a clean dataset
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


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2]]))).round()
        val_targ = self.validation_data[3]
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
        # print("\nVAL_ACC:\n", validation_results)
        # print("\n\n")
        # print("F1:\n", f1_results)

        # Save the model for another time
        # save_model(self.model, save_path)
        return


metrics = Metrics()


# define the model
def define_model(length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=1, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    # merge
    merged = concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    # compile
    adam = optimizers.Adam(lr=0.0005, decay=0.01, beta_1=0.92, beta_2=0.9992)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# load training dataset
trainLines, trainLabels = get_data(n=20000)

# create tokenizer
tokenizer = create_tokenizer(trainLines)

# calculate max document length
length = max_length(trainLines)

# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# encode data
trainX = encode_text(tokenizer, trainLines, length)

trainX, testX, trainLabels, testLabels = train_test_split(trainX, trainLabels, test_size=0.10)

# define, fit and save model
model = define_model(length, vocab_size)
# fit model
trainX = np.array(trainX)
trainLabels = np.array(trainLabels)
testX = np.array(testX)
testLabels = np.array(testLabels)

# FIT
class_weight = {0: 1.0,
                1: 2.0}
history = model.fit(x=[trainX, trainX, trainX], y=array(trainLabels),
                    validation_data=([testX, testX, testX], array(testLabels)),
                    nb_epoch=30, batch_size=64, callbacks=[metrics], class_weight=class_weight)

# Evaluate
# labels_pred = model.predict_classes(x=X_test)
loss, accuracy = model.evaluate(x=[testX, testX, testX], y=testLabels, verbose=0)
print("\bTest accuracy = " + str(round(accuracy * 100, 2)) + "%")

print("TRAIN:", history.history['acc'])
print("\n")
print("TEST:", history.history['val_acc'])
print("\n")
print("F1:", f1_results)
print("Max F1 was", max(f1_results), "at epoch", f1_results.index(max(f1_results))+1)


# Alt-embedding
# embedding_matrix = get_glove_matrix_from_dump()
# e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
#                   input_length=max_len, trainable=False)