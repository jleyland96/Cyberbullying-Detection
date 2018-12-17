import csv
from collections import defaultdict
import numpy as np
import random
import sklearn
import sklearn.metrics as sm
from sklearn import svm, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


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
    print("\nGETTING DATA")

    with open('cleaned_text_messages.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:

            if line_count == 0:
                print(row)
            else:
                label_bullying = int(row[0])
                text_message = row[1]

                if line_count % 200 == 0:
                    print(line_count)

                X.append(text_message)
                y.append(label_bullying)
            line_count += 1

    print("processed", line_count-1, "comments\n")
    return X, y


corpus, y = get_data()
print("vectorising...")
vec = CountVectorizer()

print("splitting...")
X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.20)
corpus_fit_transform = vec.fit_transform(corpus)

newVec = CountVectorizer(vocabulary=vec.vocabulary_)
X_train = newVec.fit_transform(X_train).toarray()
X_test = newVec.fit_transform(X_test).toarray()
print(X_train.shape)
print(X_test.shape)

# Repeat positive examples if you want to
X_train, y_train = repeat_positives(X_train, y_train, 2)

# loop through classifiers
for current_clf in range(0, 9):

    # TRAIN
    print("\ntraining...")

    # CHOOSE CLASSIFIER
    if current_clf == 0:
        print("Logistic regression...")
        clf = sklearn.linear_model.LogisticRegression(penalty="l2", max_iter=1000, solver="liblinear")
    elif current_clf == 1:
        print("Random Forest...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=4)
    elif current_clf == 2:
        print("Bernoulli NB...")
        clf = BernoulliNB()
    elif current_clf == 3:
        print("Gaussian NB...")
        clf = GaussianNB()
    elif current_clf == 4:
        print("Multinomial NB...")
        clf = MultinomialNB()
    elif current_clf == 5:
        print("KNN 3...")
        clf = KNeighborsClassifier(3)
    elif current_clf == 6:
        print("Adaboost...")
        clf = AdaBoostClassifier()
    elif current_clf == 7:
        print("SVM linear...")
        clf = svm.SVC(gamma="auto")
    elif current_clf == 8:
        print("Decision Trees...")
        clf = tree.DecisionTreeClassifier()

    # FIT
    clf.fit(X_train, y_train)

    # PREDICT
    print("\nevaluating")
    y_pred = clf.predict(X_test)
    print(y_pred[:20])

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
