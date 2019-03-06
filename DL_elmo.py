import tensorflow as tf
import pandas as pd
import csv
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import numpy as np

validation_results = []
f1_results = []


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


# Pre-process data
def preprocess_data():
    # GET ALL THE DATA
    X, y = get_data("cleaned_tweets_16k.csv")
    data = {}
    data['sentence'] = X
    data['label'] = y
    df = pd.DataFrame.from_dict(data)
    print(df.head())

    # SPLIT INTO TRAIN/TEST
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    test_df = df[~msk]
    print(len(train_df))
    print(len(test_df))

    # PUT INTO VALID INPUT FORM
    train_text = train_df['sentence'].tolist()
    train_text = [' '.join(t.split()[0:30]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_df['label'].tolist()
    test_text = test_df['sentence'].tolist()
    test_text = [' '.join(t.split()[0:150]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = test_df['label'].tolist()

    return train_text, test_text, train_label, test_label


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


# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)
class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# Function to build model
def build_model(): 
  input_text = layers.Input(shape=(1,), dtype="string")
  embedding = ElmoEmbeddingLayer()(input_text)
  dense = layers.Dense(64, activation='relu')(embedding)
  pred = layers.Dense(1, activation='sigmoid')(dense)

  model = Model(inputs=[input_text], outputs=pred)

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  
  return model


# Function where we fit the model
def fit_model(train_text, test_text, train_label, test_label):
    # FIT THE MODEL
    model = build_model()
    history = model.fit(train_text,
                        train_label,
                        validation_data=(test_text, test_label),
                        epochs=1,
                        batch_size=128)
    return model, history


# RE-LOAD THE MODEL
def reload_model():
    model = None
    model = build_model()
    model.load_weights('ElmoModel.h5')
    return model


# Print results after training
def print_results(history):
    print("TRAIN:", list(np.round(history.history['acc'], 4)), "\n")
    print("TEST:", list(np.round(history.history['val_acc'], 4)), "\n")
    print("LOSS:", list(np.round(history.history['loss'], 4)), "\n")
    print("Max F1 was", max(f1_results), "at epoch", f1_results.index(max(f1_results)) + 1, "\n")
    print("F1:", f1_results)


if __name__ == "__main__":
    # GET THE DATA IN THE RIGHT FORMAT
    train_text, test_text, train_label, test_label = preprocess_data()

    # FIT THE MODEL
    model, history = fit_model(train_text, test_text, train_label, test_label)
    print_results(history)

    # SAVE THE MODEL WEIGHTS
    model.save('ElmoModel.h5')

    # RELOAD THE MODEL IF NEEDED
    model = reload_model()


