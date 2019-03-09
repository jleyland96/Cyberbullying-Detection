import csv
import random
from collections import defaultdict, OrderedDict
from operator import add
import numpy as np
import keras
import sklearn
import sklearn.metrics as sm
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def most_common(lst):
    return max(set(lst), key=lst.count)


def generate_all_char_trigrams():
    bigram_dict = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(0, 26):
        for j in range(0, 26):
            for k in range(0, 26):
                gram = str(alphabet[i])+str(alphabet[j])+str(alphabet[k])
                bigram_dict[gram] = 0
    return OrderedDict(sorted(bigram_dict.items()))


def get_data(filename):
    X = []
    y = []
    print("\nGETTING DATA FROM", filename)

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                print(row)
            else:
                X.append(row[1])
                y.append(int(row[0]))
            line_count += 1

    print("processed", line_count-1, "comments\n")
    return X, y


def get_ngram_data(X_train, X_test):
    X_train_temp = []
    X_test_temp = []
    print("\nGETTING DATA TRIGRAMS")

    # compute all possible n-grams and create a base dictionary for counting them
    global_grams = generate_all_char_trigrams()

    # GET TRIGRAMS FOR TRAINING DATA
    for x in X_train:
        # current features
        temp_x = []
        this_bigram_dict = global_grams.copy()

        # split text messages into a list of its ngrams
        ngram = [x[j:j+3] for j in range(len(x)-(2))]

        # count occurences of each character ngram
        for gram in ngram:
            if gram in this_bigram_dict:
                this_bigram_dict[gram] += 1

        # create feature vector for this instance (take just the values)
        for key in this_bigram_dict:
            temp_x.append(this_bigram_dict[key])

        X_train_temp.append(temp_x)

        if len(X_train_temp) % 100 == 0:
            print(len(X_train_temp))

        del this_bigram_dict

    # GET TRIGRAMS FOR TRAINING DATA
    for x in X_test:
        # current features
        temp_x = []
        this_bigram_dict = global_grams.copy()

        # split text messages into a list of its ngrams
        ngram = [x[j:j + 3] for j in range(len(x) - (2))]

        # count occurences of each character ngram
        for gram in ngram:
            if gram in this_bigram_dict:
                this_bigram_dict[gram] += 1

        # create feature vector for this instance (take just the values)
        for key in this_bigram_dict:
            temp_x.append(this_bigram_dict[key])

        X_test_temp.append(temp_x)

        del this_bigram_dict

        if len(X_test_temp) % 100 == 0:
            print(len(X_test_temp))

    return X_train_temp, X_test_temp


def get_term_count_data(corpusX, X_train, X_test):
    print("Term Count...")
    vec = CountVectorizer()

    # Fit_transform on all of the data
    _ = vec.fit_transform(corpusX)

    # apply this vectorizer on the train and test set separately
    newVec = CountVectorizer(vocabulary=vec.vocabulary_)
    X_train_term_count = newVec.fit_transform(X_train).toarray()
    X_test_term_count = newVec.fit_transform(X_test).toarray()

    print(X_train_term_count.shape)
    print(X_test_term_count.shape)
    print()

    return X_train_term_count, X_test_term_count


def get_term_freq_data(use_idf, corpus, X_train, X_test):
    # Indicates if we are using TF or TF-IDF
    USE_IDF = use_idf
    print("Using IDF: " + str(USE_IDF))

    # GET THE DATA
    print("vectorising...")
    vec = TfidfVectorizer(min_df=0.0001, max_df=1.0)

    _ = vec.fit_transform(corpus)

    newVec = TfidfVectorizer(vocabulary=vec.vocabulary_, use_idf=USE_IDF)
    X_train_temp = newVec.fit_transform(X_train).toarray()
    X_test_temp = newVec.fit_transform(X_test).toarray()
    print(X_train_temp.shape)
    print(X_test_temp.shape)
    print()

    return X_train_temp, X_test_temp


def two_class_ensemble():
    file = "cleaned_tweets_16k.csv"

    # get the raw data and split into the training and test sets
    corpusX, corpusY = get_data(file)
    X_train, X_test, y_train, y_test = train_test_split(corpusX, corpusY, test_size=0.2)

    # get term_count features for X
    X_train_term_count, X_test_term_count = get_term_count_data(corpusX, X_train, X_test)
    # get TF features for X
    X_train_tf, X_test_tf = get_term_freq_data(False, corpusX, X_train, X_test)
    # get TF-IDF features for X
    X_train_tf_idf, X_test_tf_idf = get_term_freq_data(True, corpusX, X_train, X_test)

    # Fit all of my classifiers
    print("Classifier 1")
    clf1 = BernoulliNB().fit(X_train_term_count, y_train)
    print("Classifier 2")
    clf2 = MultinomialNB().fit(X_train_term_count, y_train)
    print("Classifier 3")
    clf3 = LogisticRegression(penalty="l2", solver="liblinear", max_iter=200).fit(X_train_term_count, y_train)
    print("Classifier 4")
    clf4 = tree.DecisionTreeClassifier().fit(X_train_term_count, y_train)

    print("Classifier 5")
    clf5 = BernoulliNB().fit(X_train_tf, y_train)
    print("Classifier 6")
    clf6 = tree.DecisionTreeClassifier().fit(X_train_tf, y_train)

    print("Classifier 7")
    clf7 = BernoulliNB().fit(X_train_tf_idf, y_train)
    print("Classifier 8")
    clf8 = LogisticRegression(penalty="l2", solver="liblinear", max_iter=200).fit(X_train_tf_idf, y_train)

    # Let the classifiers make their votes
    y_preds = []
    y_preds.append(clf1.predict(X_train_term_count))
    y_preds.append(clf2.predict(X_train_term_count))
    y_preds.append(clf3.predict(X_train_term_count))
    y_preds.append(clf4.predict(X_train_term_count))
    y_preds.append(clf5.predict(X_train_tf))
    y_preds.append(clf6.predict(X_train_tf))
    y_preds.append(clf7.predict(X_train_tf_idf))
    y_preds.append(clf8.predict(X_train_tf_idf))
    print(y_test)

    # Gather the votes and work out the majority for each example
    final_preds = []
    for pred in range(len(y_test)):
        current_y_preds = []
        # Gather the votes for this test example
        for i in range(8):
            current_y_preds.append(y_preds[i][pred])
        # Our prediction is the most common item in this list
        print(current_y_preds)
        final_preds.append(most_common(current_y_preds))
    print(final_preds)

    # Evaluate
    print("confusion matrix:\n", sm.confusion_matrix(y_test, final_preds))
    print("accuracy:", round(sm.accuracy_score(y_test, final_preds), 4))
    print("f1 score weighted:", round(sm.f1_score(y_test, final_preds), 4))
    print("f1 score micro   :", round(sm.f1_score(y_test, final_preds), 4))

    # ----- MY BEST CLASSIFIERS, 0.70 minimum -----
    # term_count        - NB Bernoulli
    # term_count        - NB Multinomial
    # term_count        - Log_reg
    # term_count        - Decision Trees
    # TF                - NB Bernoulli
    # TF                - Decision Trees
    # TF-IDF            - NB Bernoulli
    # TF-IDF            - Log_reg

    pass


def three_class_ensemble():
    file = "cleaned_tweets_16k_3class.csv"

    # get the raw data and split into the training and test sets
    corpusX, corpusY = get_data(file)
    X_train, X_test, y_train, y_test = train_test_split(corpusX, corpusY, test_size=0.2)

    # get term_count features for X
    X_train_term_count, X_test_term_count = get_term_count_data(corpusX, X_train, X_test)
    # get TF features for X
    X_train_tf, X_test_tf = get_term_freq_data(False, corpusX, X_train, X_test)
    # get TF-IDF features for X
    X_train_tf_idf, X_test_tf_idf = get_term_freq_data(True, corpusX, X_train, X_test)
    # get the character trigrams features for X
    X_train_trigrams, X_test_trigrams = get_ngram_data(X_train, X_test)

    # Fit all of my classifiers
    print("Classifier 1")
    clf1 = BernoulliNB().fit(X_train_term_count, y_train)
    print("Classifier 2")
    clf2 = MultinomialNB().fit(X_train_term_count, y_train)
    print("Classifier 3")
    clf3 = LogisticRegression(penalty="l2", solver="liblinear", max_iter=200).fit(X_train_term_count, y_train)
    print("Classifier 4")
    clf4 = tree.DecisionTreeClassifier().fit(X_train_term_count, y_train)
    print("Classifier 5")
    clf5 = GradientBoostingClassifier(200).fit(X_train_term_count, y_train)

    print("Classifier 6")
    clf6 = BernoulliNB().fit(X_train_tf, y_train)
    print("Classifier 7")
    clf7 = LogisticRegression(penalty="l2", solver="liblinear", max_iter=200).fit(X_train_tf, y_train)
    print("Classifier 8")
    clf8 = AdaBoostClassifier().fit(X_train_tf, y_train)
    print("Classifier 9")
    clf9 = GradientBoostingClassifier(200).fit(X_train_tf, y_train)

    print("Classifier 10")
    clf10 = BernoulliNB().fit(X_train_tf_idf, y_train)
    print("Classifier 11")
    clf11 = LogisticRegression(penalty="l2", solver="liblinear", max_iter=200).fit(X_train_tf_idf, y_train)
    print("Classifier 12")
    clf12 = GradientBoostingClassifier(200).fit(X_train_tf_idf, y_train)

    print("Classifier 13")
    clf13 = LogisticRegression(penalty="l2", solver="liblinear", max_iter=200).fit(X_train_trigrams, y_train)
    print("Classifier 14")
    clf14 = svm.SVC(C=10, kernel="rbf", gamma=0.001).fit(X_train_trigrams, y_train)
    print("Classifier 15")
    clf15 = GradientBoostingClassifier(200).fit(X_train_trigrams, y_train)

    # Let the classifiers make their votes
    y_preds = []
    y_preds.append(clf1.predict(X_train_term_count))
    y_preds.append(clf2.predict(X_train_term_count))
    y_preds.append(clf3.predict(X_train_term_count))
    y_preds.append(clf4.predict(X_train_term_count))
    y_preds.append(clf5.predict(X_train_term_count))
    y_preds.append(clf6.predict(X_train_tf))
    y_preds.append(clf7.predict(X_train_tf))
    y_preds.append(clf8.predict(X_train_tf))
    y_preds.append(clf9.predict(X_train_tf))
    y_preds.append(clf10.predict(X_train_tf_idf))
    y_preds.append(clf11.predict(X_train_tf_idf))
    y_preds.append(clf12.predict(X_train_tf_idf))
    y_preds.append(clf13.predict(X_train_trigrams))
    y_preds.append(clf14.predict(X_train_trigrams))
    y_preds.append(clf15.predict(X_train_trigrams))
    print(y_test)

    # Gather the votes and work out the majority for each example
    final_preds = []
    for pred in range(len(y_test)):
        current_y_preds = []
        # Gather the votes for this test example
        for i in range(15):
            current_y_preds.append(y_preds[i][pred])
        # Our prediction is the most common item in this list
        print(current_y_preds)
        final_preds.append(most_common(current_y_preds))
    print(final_preds)

    # Evaluate
    print("confusion matrix:\n", sm.confusion_matrix(y_test, final_preds))
    print("accuracy:", round(sm.accuracy_score(y_test, final_preds), 4))
    print("f1 score weighted:", round(sm.f1_score(y_test, final_preds, average='weighted'), 4))
    print("f1 score micro   :", round(sm.f1_score(y_test, final_preds, average='micro'), 4))

    # ----- MY BEST CLASSIFIERS, 0.80 minimum -----
    # term_count    - NB_Bernoulli
    # term_count    - NB_Multinomial
    # term_count    - Log_reg
    # term_count    - Decision trees
    # term_count    - GBC (200)
    # TF            - NB_Bernoulli
    # TF            - Log_reg
    # TF            - Adaboost
    # TF            - GBC (200)
    # TFIDF         - NB_Bernoulli
    # TFIDF         - Log_reg
    # TFIDF         - GBC (200)
    # Trigrams      - Log_reg
    # Trigrams      - SVM
    # Trigrams      - GBC (200)


three_class_ensemble()
