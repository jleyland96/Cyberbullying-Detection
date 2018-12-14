import csv
import numpy as np
import sklearn
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import keras
from collections import defaultdict


def get_data():
    X = []
    y = []
    print("\ngetting data...")

    with open('cleaned_text_messages.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                print(row)
            else:
                label_bullying = int(row[0])
                text_message = row[1]

                X.append(text_message)
                y.append(label_bullying)
            line_count += 1

    print("processed", line_count-1, "comments")
    return X, y


# SPLIT COMMENTS/SENTENCES
comments, y = get_data()
num_comments = len(comments)
print("splitting data...")
word_arrays = []
for s in comments:
    word_arrays.append(s.split(' '))


# GLOVE. Create dictionary where keys are words and the values are the vectors for the words
print("getting GLOVE embeddings size 300...")
file = open('glove.6B/glove.6B.300d.txt', "r").readlines()
gloveDict = {}
for line in file:
    info = line.split(' ')
    key = info[0]
    vec = []
    for elem in info[1:]:
        vec.append(elem.rstrip())
    gloveDict[key] = vec
print(len(gloveDict), "words in the GLOVE dictionary\n")

# VECTORISE WORDS
print("converting comments to lists of vectors...")
word_vectors = []
for sentence in word_arrays:
    temp = []
    for word in sentence:
        if word in gloveDict:
            temp.append(gloveDict[word])
    word_vectors.append(temp)

# PADDING
MAX_LEN = 40
print("padding vectors to maxlen =", MAX_LEN, "...")
padded_word_vecs = np.array(keras.preprocessing.sequence.pad_sequences(word_vectors, padding='pre', maxlen=MAX_LEN, dtype='float32'))
padded_word_vecs = padded_word_vecs.reshape((num_comments, -1))

print("DONE PRE-PROCESSING\n")

# CLASSIFYING
print("splitting...")
X_train, X_test, y_train, y_test = train_test_split(padded_word_vecs, y, test_size=0.20)
# TRAIN_PROPORTION = 0.8
# train_count = int(TRAIN_PROPORTION*num_comments)
# X_train = padded_word_vecs[:train_count]
# y_train = y[:train_count]
# X_test = padded_word_vecs[train_count:]
# y_test = y[train_count:]


# # Scale the feature values from -1 to 1, as this speeds up training time
# scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_test = scaling.transform(X_test)


# CYCLE THROUGH THE CLASSIFIERS
for current_clf in range(0, 9):
    if current_clf == 0:
        print("LOGISTIC REGRESSION")
        clf = sklearn.linear_model.LogisticRegression(penalty="l2", max_iter=100, solver="liblinear")
    elif current_clf == 1:
        print("RANDOM FOREST...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=4)
    elif current_clf == 2:
        print("BERNOULLI NB...")
        clf = BernoulliNB()
    elif current_clf == 3:
        print("GAUSSIAN NB...")
        clf = GaussianNB()
    elif current_clf == 4:
        print("MULTINOMIAL NB")
        clf = MultinomialNB()
    elif current_clf == 5:
        print("KNN 3...")
        clf = KNeighborsClassifier(3)
    elif current_clf == 6:
        print("ADABOOST...")
        clf = AdaBoostClassifier()
    elif current_clf == 7:
        print("SVM LINEAR...")
        clf = svm.SVC(gamma='auto')
    elif current_clf == 8:
        print("Decision Trees...")
        clf = tree.DecisionTreeClassifier()

    # FIT
    clf.fit(X_train, y_train)

    # PREDICT
    print("evaluating")
    y_pred = clf.predict(X_test)

    # EVALUATE
    print("confusion matrix:", sm.confusion_matrix(y_test, y_pred))
    print("accuracy:", sm.accuracy_score(y_test, y_pred))

    # if the metrics are well defined
    if np.count_nonzero(y_pred) > 0:
        print("recall:", sm.recall_score(y_test, y_pred))
        print("precision:", sm.precision_score(y_test, y_pred))
        print("f1 score:", sm.f1_score(y_test, y_pred))
        print("\n\n")
    else:
        print("No True predictions made\n\n")


