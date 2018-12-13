import csv
import sklearn.metrics as sm
import matplotlib.pyplot as plt


def naive_2_class_classifier(th):
    with open('cleaned_text_messages.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        gold_labels = []
        prediction = []

        for row in csv_reader:
            if line_count == 0:
                print(row)
            else:
                naughty_words_frequency = float(row[4])
                label_bullying = int(row[0])

                # make prediction
                if naughty_words_frequency < th:
                    prediction.append(0)
                else:
                    prediction.append(1)

                # gold standard data
                gold_labels.append(label_bullying)

            line_count += 1
    return prediction, gold_labels


# 0.014012 is best so far
threshold = 0.0001
x_acc = []
y_acc = []
max_acc = 0
max_threshold = 0

while threshold < 1.0:
    print("\nClassifying with threshold:", threshold)
    pred, gold = naive_2_class_classifier(threshold)

    # evaluate
    print("Accuracy: ", "{:.4f}".format(sm.accuracy_score(gold, pred)))
    print("Recall: ", "{:.4f}".format(sm.recall_score(gold, pred)))
    print("Precision: ", "{:.4f}".format(sm.precision_score(gold, pred)))
    print("F1 score: ", "{:.4f}".format(sm.f1_score(gold, pred)))

    x_acc += [threshold]
    y_acc += [sm.accuracy_score(gold, pred)]

    # update best threshold yet
    if y_acc[-1] > max_acc:
        max_acc = y_acc[-1]
        max_threshold = threshold

    threshold += 0.001

plt.plot(x_acc, y_acc, 'r')
plt.title("Accuracy based on different norm thresholds")
plt.savefig("Screenshots/naive classifier graph 2")
plt.show()

print("\nbest accuracy:", max_acc, "achieved with threshold:", max_threshold)