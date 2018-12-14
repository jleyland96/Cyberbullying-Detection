import csv
from collections import defaultdict, OrderedDict
import numpy as np
import sklearn
import sklearn.metrics as sm
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


def generate_all_char_bigrams():
    bigram_dict = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(0, 26):
        for j in range(0, 26):
            gram = str(alphabet[i])+str(alphabet[j])
            bigram_dict[gram] = 0

    return OrderedDict(sorted(bigram_dict.items()))


def generate_all_char_trigrams():
    bigram_dict = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(0, 26):
        for j in range(0, 26):
            for k in range(0, 26):
                gram = str(alphabet[i])+str(alphabet[j])+str(alphabet[k])
                bigram_dict[gram] = 0
    return OrderedDict(sorted(bigram_dict.items()))


def get_data(ngram_size):
    X = []
    y = []
    print("\nGETTING DATA - " + str(ngram_size) + "-grams")

    # compute all possible n-grams and create a base dictionary for counting them
    if ngram_size == 2:
        global_grams = generate_all_char_bigrams()
    if ngram_size == 3:
        global_grams = generate_all_char_trigrams()

    # READ CSV
    with open('cleaned_text_messages.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                print(row)
            else:
                if line_count % 200 == 0:
                    print(str(line_count) + " ngrams computed")

                label_bullying = int(row[0])
                text_message = row[1]

                # current features
                temp_x = []
                this_bigram_dict = global_grams.copy()

                # split text messages into a list of its ngrams
                ngram = [text_message[j:j+ngram_size] for j in range(len(text_message)-(ngram_size-1))]

                # TODO: change this to 'count' so we get better performance
                # count occurences of each character ngram
                for gram in ngram:
                    if gram in this_bigram_dict:
                        this_bigram_dict[gram] += 1

                # create feature vector for this instance (take just the values)
                for key in this_bigram_dict:
                    temp_x.append(this_bigram_dict[key])

                X.append(temp_x)
                y.append(label_bullying)

                del this_bigram_dict
            line_count += 1
    print("processed", line_count-1, "comments\n")
    return X, y


x, y = get_data(2)

# SPLIT
print("splitting...")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# loop through classifiers
for current_clf in range(0, 9):
    # TRAIN
    print("\ntraining...")

    # CHOOSE CLASSIFIER
    if current_clf == 0:
        print("Logistic regression...")
        clf = sklearn.linear_model.LogisticRegression(penalty="l2", max_iter=100, solver="liblinear")
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
    print("fitting...")
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
