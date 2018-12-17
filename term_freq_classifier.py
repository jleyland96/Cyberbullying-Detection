import csv
from collections import defaultdict
import numpy as np
import sklearn
import sklearn.metrics as sm
from sklearn import svm, tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


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

                # if getting raw data just return the comments themselves
                X.append(text_message)
                y.append(label_bullying)
            line_count += 1

    print("processed", line_count-1, "comments\n")
    return X, y


# Indicates if we are using TF or TF-IDF
USE_IDF = False
print("Using IDF: " + str(USE_IDF))

# GET THE DATA
corpus, y = get_data()
print("vectorising...")
vec = TfidfVectorizer(min_df=0.0001, max_df=1.0)

X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.20)
corpus_fit_transform = vec.fit_transform(corpus)

newVec = TfidfVectorizer(vocabulary=vec.vocabulary_, use_idf=USE_IDF)
X_train = newVec.fit_transform(X_train).toarray()
X_test = newVec.fit_transform(X_test).toarray()
print(X_train.shape)
print(X_test.shape)
print()

print("BEFORE REPEATS")
print(len(X_train[0]), "features")
print(len(X_test[0]))
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
print()

# Repeat the positive examples in the training dataset twice to avoid over-fitting to negative examples
X_train, y_train = repeat_positives(X_train, y_train, repeats=2)

print("AFTER REPEATS")
print(len(X_train[0]), "features")
print(len(X_test[0]))
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
print()

# CYCLE THROUGH THE CLASSIFIERS
for current_clf in range(0, 9):
    if current_clf == 0:
        print("LOGISTIC REGRESSION")
        clf = sklearn.linear_model.LogisticRegression(penalty="l2", max_iter=100, solver="liblinear")
    elif current_clf == 1:
        print("RANDOM FOREST...")
        clf = RandomForestClassifier(n_estimators=1000, max_depth=12)
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
