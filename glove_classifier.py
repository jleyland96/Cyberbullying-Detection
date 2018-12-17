import csv
import numpy as np
import sklearn
import random
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import keras
from collections import defaultdict


def shuffle_data(X, y):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
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

# Repeat the positive examples in the training dataset if you want
X_train, y_train = repeat_positives(X_train, y_train, 2)


# CYCLE THROUGH THE CLASSIFIERS
for current_clf in range(0, 9):
    if current_clf == 0:
        print("LOGISTIC REGRESSION")
        clf = sklearn.linear_model.LogisticRegression(penalty="l2", max_iter=100, solver="liblinear")
    elif current_clf == 1:
        print("RANDOM FOREST...")
        clf = RandomForestClassifier(n_estimators=300, max_depth=8)
    elif current_clf == 2:
        print("BERNOULLI NB...")
        clf = BernoulliNB()
    elif current_clf == 3:
        print("GAUSSIAN NB...")
        clf = GaussianNB()
    elif current_clf == 4:
        print("KNN 3...")
        clf = KNeighborsClassifier(3)
    elif current_clf == 5:
        print("ADABOOST...")
        clf = AdaBoostClassifier()
    elif current_clf == 6:
        print("SVM LINEAR...")
        clf = svm.SVC(gamma='auto')
    elif current_clf == 7:
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


