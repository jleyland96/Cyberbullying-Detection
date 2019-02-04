from pickle import load
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


# define the model
def define_model(length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding3)
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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

# define, fit and save model
model = define_model(length, vocab_size)
# fit model
model.fit([trainX,trainX, trainX], array(trainLabels), epochs=10, batch_size=32)


# Alt-embedding
# embedding_matrix = get_glove_matrix_from_dump()
# e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
#                   input_length=max_len, trainable=False)