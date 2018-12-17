import csv
import random
from collections import defaultdict, OrderedDict
import numpy as np
import keras
import sklearn
import sklearn.metrics as sm
from sklearn.model_selection import GridSearchCV
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


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
                X.append(row[1])
                y.append(int(row[0]))
            line_count += 1

    print("processed", line_count-1, "comments\n")
    return X, y


def get_ngram_data(ngram_size):
    X = []
    y = []
    print("\nGETTING DATA - " + str(ngram_size) + "-grams")

    # compute all possible n-grams and create a base dictionary for counting them
    if ngram_size == 2:
        global_grams = generate_all_char_bigrams()
    else:
        global_grams = generate_all_char_trigrams()

    # READ CSV
    with open('cleaned_text_messages.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                print(row)
            else:
                if line_count % 500 == 0:
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

    # shuffle the data so that it is randomised
    X, y = shuffle_data(X, y)

    # SPLIT
    print("splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    return X_train, X_test, y_train, y_test


def get_term_count_data():
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
    print()

    return X_train, X_test, y_train, y_test


def get_term_freq_data(use_idf):
    # Indicates if we are using TF or TF-IDF
    USE_IDF = use_idf
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

    return X_train, X_test, y_train, y_test


def get_glove_data():
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
    padded_word_vecs = np.array(
        keras.preprocessing.sequence.pad_sequences(word_vectors, padding='pre', maxlen=MAX_LEN, dtype='float32'))
    padded_word_vecs = padded_word_vecs.reshape((num_comments, -1))

    print("DONE PRE-PROCESSING\n")

    # CLASSIFYING
    print("splitting...")
    X_train, X_test, y_train, y_test = train_test_split(padded_word_vecs, y, test_size=0.20)

    return X_train, X_test, y_train, y_test


print("WITHOUT REPEATS - TUNING BUT NOT ON LINEAR REGRESSION")
for dataset_choice in ["glove", "term_count", "term_freq", "term_freq_idf", "bigrams", "trigrams"]:

    # Get the right dataset (Glove features, term count, term freq, term freq idf, bigrams, trigrams)
    if dataset_choice == "glove":
        X_train, X_test, y_train, y_test = get_glove_data()
    elif dataset_choice == "term_count":
        X_train, X_test, y_train, y_test = get_term_count_data()
    elif dataset_choice == "term_freq":
        X_train, X_test, y_train, y_test = get_term_freq_data(use_idf=False)
    elif dataset_choice == "term_freq_idf":
        X_train, X_test, y_train, y_test = get_term_freq_data(use_idf=True)
    elif dataset_choice == "bigrams":
        X_train, X_test, y_train, y_test = get_ngram_data(ngram_size=2)
    else:
        X_train, X_test, y_train, y_test = get_ngram_data(ngram_size=3)

    # Repeat the positive examples in the training dataset twice to avoid over-fitting to negative examples
    # X_train, y_train = repeat_positives(X_train, y_train, repeats=2)

    # loop through classifiers
    for current_clf in range(0, 10):
        # TRAIN
        print("\ntraining on dataset", dataset_choice, "...")
        grid_searching = False

        # CHOOSE CLASSIFIER
        if current_clf == 0:
            print("Logistic regression...")
            # grid_searching = True
            # param_grid = {'max_iter': [100, 300], 'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag']}
            # clf = GridSearchCV(sklearn.linear_model.LogisticRegression(), param_grid, cv=3)
            clf = sklearn.linear_model.LogisticRegression(penalty="l2", max_iter=300, solver="liblinear")
        elif current_clf == 1:
            print("Random Forest...")
            # grid_searching = True
            # param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [3, 6, 10, 12]}
            # clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
            clf = RandomForestClassifier(n_estimators=300, max_depth=12)
        elif current_clf == 2:
            print("Bernoulli NB...")
            clf = BernoulliNB()
        elif current_clf == 3:
            print("Gaussian NB...")
            clf = GaussianNB()
        elif current_clf == 4 and not(dataset_choice == "glove"):
            print("Multinomial NB...")
            clf = MultinomialNB()
        elif current_clf == 5:
            print("KNN 3...")
            # grid_searching = True
            # param_grid = {'n_neighbors': [1, 3]}
            # clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
            clf = KNeighborsClassifier(n_neighbors=3)
        elif current_clf == 6:
            print("Adaboost...")
            clf = AdaBoostClassifier()
        elif current_clf == 7:
            print("SVM...")
            # grid_searching = True
            # param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
            #               {'kernel': ['linear'], 'C': [1, 10, 100]}]
            # clf = GridSearchCV(svm.SVC(), param_grid, cv=3)
            clf = svm.SVC(gamma="auto")
        elif current_clf == 8:
            print("Decision Trees...")
            clf = tree.DecisionTreeClassifier()
        else:
            print("Gradient boosted classifier...")
            clf = GradientBoostingClassifier(n_estimators=100)

        # FIT
        print("fitting...")
        clf = clf.fit(X_train, y_train)

        # If we did a grid search, then we want to print what the best estimator was
        # if grid_searching:
        #     print("Best estimator found by grid search:")
        #     print(clf.best_estimator_)

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
